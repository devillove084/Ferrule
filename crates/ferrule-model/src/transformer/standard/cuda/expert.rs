//! Optional host-control EP seam. Expert matmuls/SiLU always run on CUDA.

use std::collections::BTreeMap;
use std::rc::Rc;

use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{ParallelRankId, Result};

use crate::execution::ExecutionPrecisionPolicy;
use crate::transformer::expert_parallel::{
    ExpertDispatchLimits, ExpertDispatchPlan, ExpertPlacement, ExpertResult, ExpertResultExecutor,
    ExpertToken, ExpertWorker, RoutedSwiGluExecutor, RoutedSwiGluRequest,
};
use crate::transformer::{
    ExpertAvailability, ExpertProvider, HostRows, OperatorProgress, Rows, RowsDType, RowsShape,
    StandardDecoderOperators,
};

use super::{CudaStandardDecoderOperators, cuda_error, device_rows};

/// Host messages carry only unweighted expert results. Validation/ordering uses
/// the same dispatch plan as CPU EP, but weighting and reduction stay on CUDA.
/// The caller must supply GPU expert owners; no local expert fallback is used.
pub struct CudaExpertParallelRoutedExecutor<'a> {
    operators: &'a mut CudaStandardDecoderOperators,
    members: Vec<ParallelRankId>,
    placement: &'a ExpertPlacement,
    limits: ExpertDispatchLimits,
    results: &'a mut dyn ExpertResultExecutor,
}

impl<'a> CudaExpertParallelRoutedExecutor<'a> {
    pub fn new(
        operators: &'a mut CudaStandardDecoderOperators,
        members: Vec<ParallelRankId>,
        placement: &'a ExpertPlacement,
        limits: ExpertDispatchLimits,
        results: &'a mut dyn ExpertResultExecutor,
    ) -> Self {
        Self {
            operators,
            members,
            placement,
            limits,
            results,
        }
    }
}

impl RoutedSwiGluExecutor for CudaExpertParallelRoutedExecutor<'_> {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        check_active(request.context.transaction)?;
        self.operators.f32(request.input)?;
        if request.routes.rows() != request.input.shape().rows() {
            return Err(cuda_error("routed input row count mismatch"));
        }
        // Admission precedes even the boundary download, as on the host seam.
        let plan = ExpertDispatchPlan::new(
            request.context,
            self.members.clone(),
            self.placement,
            request.routes,
            request.sequences,
            request.input.shape().width(),
            self.limits,
        )?;
        let input = self
            .operators
            .synchronous(|this| this.boundary.download(&this.ops, this.f32(request.input)?))?;
        let buckets = plan.dispatch(&input, check_active)?;
        let mut results = Vec::with_capacity(plan.token_count());
        for bucket in buckets {
            check_active(request.context.transaction)?;
            let owner = bucket.owner_rank;
            let reply = self.results.execute(bucket)?;
            check_active(request.context.transaction)?;
            plan.ordered_results(&reply, Some(owner))?;
            results.extend(reply);
        }
        let ordered = plan.ordered_results(&results, None)?;
        let values = ordered
            .iter()
            .flat_map(|r| r.output.iter().copied())
            .collect::<Vec<_>>();
        let rows = ordered
            .iter()
            .map(|r| {
                i32::try_from(r.source_row).map_err(|_| cuda_error("route row exceeds i32 ABI"))
            })
            .collect::<Result<Vec<_>>>()?;
        check_active(request.context.transaction)?;
        let output = self.operators.synchronous(|this| {
            let values = this.boundary.upload(&this.ops, &values)?;
            let rows = this.ops.upload_i32_buffer(&rows)?;
            // Frozen router weights, not values supplied by a peer. Sorted results
            // match the router's original (source_row, route_slot) order exactly.
            let weights = this.ops.upload_f32_buffer(request.routes.weights())?;
            let shape = request.input.shape();
            let mut output = this.ops.zero_f32_buffer(shape.elements())?;
            this.ops.weighted_combine_f32_into(
                &values,
                &rows,
                &weights,
                &mut output,
                shape.rows(),
                shape.width(),
            )?;
            device_rows(shape, request.arena, output)
        })?;
        check_active(request.context.transaction)?;
        Ok(OperatorProgress::Ready(output))
    }
}

/// Adapts a host-message dispatcher to GPU router outputs and resident input.
/// Only this explicit EP boundary stages activations; no attention/linear CPU
/// fallback is installed. The dispatcher must select CUDA expert workers.
pub struct CudaHostRoutedExecutor<'a> {
    staging: CudaStandardDecoderOperators,
    executor: &'a mut dyn RoutedSwiGluExecutor,
}

impl<'a> CudaHostRoutedExecutor<'a> {
    pub fn new(ops: Rc<CudaOperators>, executor: &'a mut dyn RoutedSwiGluExecutor) -> Result<Self> {
        Ok(Self {
            staging: CudaStandardDecoderOperators::new(ops, ExecutionPrecisionPolicy::f32(), &[])?,
            executor,
        })
    }
    pub fn needs_quarantine(&self) -> bool {
        self.staging.needs_quarantine()
    }
}

impl RoutedSwiGluExecutor for CudaHostRoutedExecutor<'_> {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        check_active(request.context.transaction)?;
        let input = self.staging.synchronous(|this| {
            let buffer = this.f32(request.input)?;
            let values = this.boundary.download(&this.ops, buffer)?;
            HostRows::new(request.input.shape(), RowsDType::F32, request.arena, values)
        })?;
        check_active(request.context.transaction)?;
        let input = Rows::Host(input);
        let result = self.executor.routed_swiglu(
            RoutedSwiGluRequest {
                input: &input,
                ..request
            },
            check_active,
        )?;
        check_active(request.context.transaction)?;
        match result {
            OperatorProgress::Ready(rows) => {
                self.staging.bind_rows(rows).map(OperatorProgress::Ready)
            }
            progress => Ok(progress),
        }
    }
}

/// Owner-local expert worker with a bounded prepared parameter directory.
/// Results are unweighted: the dispatch protocol applies each route weight once.
pub struct CudaExpertWorker<'a> {
    owner: ParallelRankId,
    placement: &'a ExpertPlacement,
    experts: &'a mut dyn ExpertProvider,
    operators: &'a mut CudaStandardDecoderOperators,
}

impl<'a> CudaExpertWorker<'a> {
    pub fn new(
        owner: ParallelRankId,
        placement: &'a ExpertPlacement,
        experts: &'a mut dyn ExpertProvider,
        operators: &'a mut CudaStandardDecoderOperators,
    ) -> Self {
        Self {
            owner,
            placement,
            experts,
            operators,
        }
    }
}

impl ExpertWorker for CudaExpertWorker<'_> {
    fn owner(&self) -> ParallelRankId {
        self.owner
    }

    fn compute(&mut self, tokens: &[ExpertToken]) -> Result<Vec<ExpertResult>> {
        let mut groups = BTreeMap::new();
        for token in tokens {
            if self
                .placement
                .owner(token.expert.layer, token.expert.expert)
                != Some(self.owner)
            {
                return Err(cuda_error("expert request is not assigned to this owner"));
            }
            groups
                .entry(token.expert)
                .or_insert_with(Vec::new)
                .push(token);
        }
        let mut results = Vec::with_capacity(tokens.len());
        for (expert, tokens) in groups {
            let prepared = match self.experts.expert(expert.layer, expert.expert)? {
                ExpertAvailability::Ready(prepared) => prepared,
                ExpertAvailability::Waiting => return Err(cuda_error("CUDA expert is waiting")),
                ExpertAvailability::Unsupported(reason) => return Err(cuda_error(reason)),
            };
            let width = prepared.input_width();
            if prepared.output_width() != width
                || tokens.iter().any(|token| {
                    token.payload.len() != width || token.payload.iter().any(|v| !v.is_finite())
                })
            {
                return Err(cuda_error("invalid CUDA expert payload/shape"));
            }
            let shape = RowsShape::new(tokens.len(), width)?;
            let values = tokens
                .iter()
                .flat_map(|token| token.payload.iter().copied())
                .collect();
            let input = self.operators.bind_rows(Rows::Host(HostRows::new(
                shape,
                RowsDType::F32,
                None,
                values,
            )?))?;
            let output =
                super::super::ready(self.operators.dense_swiglu(&prepared, &input, None)?)?;
            let output = self.operators.download_rows(output)?;
            if output.shape() != shape {
                return Err(cuda_error("CUDA expert output shape mismatch"));
            }
            for (token, row) in tokens.into_iter().zip(output.values().chunks_exact(width)) {
                results.push(ExpertResult::from_token(token, self.owner, row.to_vec()));
            }
        }
        Ok(results)
    }
}
