//! Synchronous, owner-local standard CUDA decoder operators.
//!
//! BF16 checkpoints are converted once to resident F32 weights. Activations,
//! KV and logits are genuinely F32; this is NOT BF16 compatibility execution.

mod hybrid;
mod recurrent;
pub use hybrid::CudaHybridModule;
mod binding;
mod boundary;
mod expert;
mod kv_binding;
mod stage;

pub use expert::{CudaExpertParallelRoutedExecutor, CudaExpertWorker, CudaHostRoutedExecutor};
pub use kv_binding::CudaStandardKvBinding;
pub use stage::CudaStandardDecoderSegment;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

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
    MoeRouterSpec, OperatorProgress, OperatorWaiting, PreparedEmbedding, PreparedLinear,
    PreparedNorm, PreparedRope, PreparedSwiGlu, RotaryEmbedding, RotaryPairing, RotaryRegion,
    RouterRoutes, Rows, RowsArenaId, RowsDType, RowsShape, StandardDecoderOperators,
    UnsupportedOperator,
};

use binding::Bindings;
use boundary::Boundary;

pub struct CudaStandardDecoderOperators {
    ops: Rc<CudaOperators>,
    deltas: BTreeMap<usize, recurrent::ResidentDelta>,
    bindings: Bindings,
    experts: BTreeMap<(usize, usize), Arc<PreparedSwiGlu>>,
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
        validate_precision(precision)?;
        let bindings = Bindings::new(parameters)?;
        let boundary = Boundary::new(&ops)?;
        Ok(Self {
            deltas: BTreeMap::new(),
            ops,
            bindings,
            experts: BTreeMap::new(),
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
                    super::PreparedFeedForwardKind::Routed { router, shared, .. } => {
                        this.bindings.linear(&this.ops, router)?;
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
            for linear in [expert.gate(), expert.up(), expert.down()] {
                this.bindings.linear(&this.ops, linear)?;
            }
            Ok(())
        })
    }

    pub fn operators(&self) -> &Rc<CudaOperators> {
        &self.ops
    }
    pub fn resident_parameter_bytes(&self) -> usize {
        self.bindings.resident_bytes()
    }
    pub fn image_generation(&self) -> u64 {
        self.bindings.generation.get()
    }
    pub fn needs_quarantine(&self) -> bool {
        self.poisoned || self.boundary.needs_quarantine()
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
        let drain = self.boundary.drain();
        let compute = self.ops.sync_stream();
        let upload = self.ops.sync_upload_stream();
        let failures = [drain, compute, upload]
            .into_iter()
            .filter_map(Result::err)
            .collect::<Vec<_>>();
        let result = match result {
            Ok(result) => result,
            Err(panic) => {
                self.poisoned = true;
                std::panic::resume_unwind(panic)
            }
        };
        if !failures.is_empty() || self.boundary.needs_quarantine() {
            self.poisoned = true;
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
        result
    }

    fn f32<'a>(&self, rows: &'a Rows) -> Result<&'a CudaF32Buffer> {
        let rows = rows.cuda()?;
        rows.validate_owner(&self.ops)?;
        rows.f32_buffer()
            .ok_or_else(|| cuda_error("standard CUDA requires F32 activations"))
    }

    fn linear_rows(
        &mut self,
        linear: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<Rows> {
        let input_buffer = self.f32(input)?;
        let weight = self.bindings.linear(&self.ops, linear)?;
        if input.shape().width() != weight.shape.1 {
            return Err(cuda_error("linear input width mismatch"));
        }
        let rows = input.shape().rows();
        let shape = RowsShape::new(rows, weight.shape.0)?;
        let mut output = self.ops.zero_f32_buffer(shape.elements())?;
        self.ops
            .linear_f32_into(&weight.handle, input_buffer, rows, &mut output)?;
        if let Some(bias) = &weight.bias {
            let ids = self.ops.upload_i32_buffer(&vec![0; rows])?;
            let bias = self
                .ops
                .gather_f32_rows(bias, &ids, rows, linear.out_features())?;
            self.ops.saxpy_into(1.0, &bias, &mut output)?;
        }
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

    fn swiglu_rows(
        &mut self,
        expert: &PreparedSwiGlu,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<Rows> {
        let gate = self.linear_rows(expert.gate(), input, arena)?;
        let up = self.linear_rows(expert.up(), input, arena)?;
        let mut product = self.ops.zero_f32_buffer(gate.shape().elements())?;
        match expert.activation_limit() {
            Some(limit) => self.ops.swiglu_f32_clamped_into(
                self.f32(&gate)?,
                self.f32(&up)?,
                &mut product,
                limit,
            )?,
            None => self
                .ops
                .swiglu_f32_into(self.f32(&gate)?, self.f32(&up)?, &mut product)?,
        }
        let product = device_rows(gate.shape(), arena, product)?;
        self.linear_rows(expert.down(), &product, arena)
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
        "cuda-standard-f32"
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
            let mut ids = this.ops.zero_i32_buffer(layout.output_elements()?)?;
            let mut weights = this.ops.zero_f32_buffer(layout.output_elements()?)?;
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
            RouterRoutes::new(layout.rows, layout.top_k, ids, weights).map(OperatorProgress::Ready)
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
            if routes.rows() != input.shape().rows()
                || routes.weights().iter().any(|w| !w.is_finite())
            {
                return Err(cuda_error("invalid routed rows/weights"));
            }
            let mut prepared = BTreeMap::new();
            let mut waiting = Vec::new();
            for id in routes.expert_ids().iter().copied().collect::<BTreeSet<_>>() {
                if let Some(expert) = this.experts.get(&(layer, id)) {
                    prepared.insert(id, Arc::clone(expert));
                    continue;
                }
                match experts.expert(layer, id)? {
                    ExpertAvailability::Ready(expert) => {
                        for linear in [expert.gate(), expert.up(), expert.down()] {
                            this.bindings.linear(&this.ops, linear)?;
                        }
                        this.experts.insert((layer, id), Arc::clone(&expert));
                        prepared.insert(id, expert);
                    }
                    ExpertAvailability::Waiting => waiting.push(id),
                    ExpertAvailability::Unsupported(reason) => {
                        return Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
                            "routed_swiglu",
                            reason,
                        )));
                    }
                }
            }
            if !waiting.is_empty() {
                return Ok(OperatorProgress::Waiting(OperatorWaiting::Experts {
                    layer,
                    experts: waiting,
                }));
            }
            let width = input.shape().width();
            let mut output = this.ops.zero_f32_buffer(input.shape().elements())?;
            for row in 0..routes.rows() {
                let ids =
                    this.ops
                        .upload_i32_buffer(&[i32::try_from(row)
                            .map_err(|_| cuda_error("route row exceeds i32 ABI"))?])?;
                let values = this.ops.gather_f32_rows(this.f32(input)?, &ids, 1, width)?;
                let row_input = device_rows(RowsShape::new(1, width)?, arena, values)?;
                let mut row_output = this.ops.zero_f32_buffer(width)?;
                let (expert_ids, weights) = routes.row(row)?;
                for (&id, &weight) in expert_ids.iter().zip(weights) {
                    let expert = prepared.get(&id).expect("prepared route");
                    if expert.output_width() != width {
                        return Err(cuda_error("routed output width mismatch"));
                    }
                    let values = this.swiglu_rows(expert, &row_input, arena)?;
                    this.ops
                        .saxpy_into(weight, this.f32(&values)?, &mut row_output)?;
                }
                this.ops
                    .scatter_add_f32_rows(&row_output, &ids, &mut output, 1, width)?;
            }
            device_rows(input.shape(), arena, output).map(OperatorProgress::Ready)
        })
    }

    fn residual(&mut self, residual: Rows, update: &Rows) -> Result<OperatorProgress<Rows>> {
        self.synchronous(|this| {
            this.f32(&residual)?;
            if residual.shape() != update.shape() {
                return Err(cuda_error("residual shape mismatch"));
            }
            let mut residual = residual.into_cuda()?;
            this.ops.residual_add_f32_in_place(
                this.f32(update)?,
                residual.f32_buffer_mut().expect("validated F32"),
            )?;
            Ok(OperatorProgress::Ready(Rows::Cuda(residual)))
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
        if !matches!(proof, Ok(true)) {
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
