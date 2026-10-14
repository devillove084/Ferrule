//! Preparation-only rank binding and allocation-free numeric GPU execution.
//! No checkpoint access, dequantization, collective, or CPU fallback at execution.
use std::{rc::Rc, sync::Arc};

use ferrule_backend::cuda::operators::linear::{
    CudaF32Buffer, CudaNumericFp8Artifact, CudaNumericFp8Workspace, CudaOperators,
    NumericFp8Layout, NumericFp8LinearPlan, NumericFp8ScaleType,
};
use ferrule_common::Result;

use super::super::{CudaStandardDecoderOperators, cuda_error};
use crate::transformer::{
    CudaNumericFp8RankBinding, NumericFp8Precision, NumericFp8RankPlan,
    PreparedNumericFp8RankLinear, Rows, hybrid_parallel::HybridRankLayout,
};

/// Numeric residency only. Caller-owned input/output buffers, allocator segment
/// slack, state/KV and other image weights remain separate admission items.
pub(in crate::transformer::standard::cuda) struct RankBindings {
    pub layout: Arc<HybridRankLayout>,
    pub limit: usize,
    pub grant_bytes: usize,
    pub linears: Vec<ResidentRankLinear>,
}

pub(in crate::transformer::standard::cuda) struct ResidentRankLinear {
    pub plan: NumericFp8RankPlan,
    artifacts: Vec<CudaNumericFp8Artifact>,
    workspace: CudaNumericFp8Workspace,
    max_rows: usize,
    scratch_bytes: usize,
    precision: NumericFp8Precision,
}

impl NumericFp8RankPlan {
    /// Exact live allocation grants for encoded segments plus ONE reusable
    /// workspace for this binding. Pure planning: works in a CUDA build without
    /// a GPU. Input/output grants are returned separately for caller admission.
    pub fn cuda_execution_budget(
        &self,
        max_rows: usize,
        scratch_bytes: usize,
        precision: NumericFp8Precision,
    ) -> Result<crate::transformer::NumericFp8RankExecutionBudget> {
        let plans = plans(self, max_rows, scratch_bytes, precision)?;
        let workspace = plans
            .iter()
            .map(|p| p.workspace_requirements().bytes as usize)
            .max()
            .unwrap();
        let workspace_grant_bytes = grant(workspace)?;
        let input_bytes = checked_product(max_rows, self.local_shape()[1], 4)?;
        let mut output_grant_bytes = 0usize;
        for plan in &plans {
            output_grant_bytes = add(
                output_grant_bytes,
                grant(checked_product(max_rows, plan.layout().n, 4)?)?,
            )?;
        }
        let resident_grant_bytes =
            add(self.budget().allocation_grant_bytes, workspace_grant_bytes)?;
        let input_grant_bytes = grant(input_bytes)?;
        Ok(crate::transformer::NumericFp8RankExecutionBudget {
            workspace_bytes: workspace,
            workspace_grant_bytes,
            resident_grant_bytes,
            input_grant_bytes,
            output_grant_bytes,
            total_grant_bytes: add(
                resident_grant_bytes,
                add(input_grant_bytes, output_grant_bytes)?,
            )?,
        })
    }
}

fn plans(
    plan: &NumericFp8RankPlan,
    rows: usize,
    scratch: usize,
    precision: NumericFp8Precision,
) -> Result<Vec<NumericFp8LinearPlan>> {
    if plan.layout().precision().is_mixed() || precision.has_bf16_model_boundaries() {
        return Err(cuda_error(
            "F32 rank numeric executor cannot execute native BF16 boundaries",
        ));
    }
    plan.reads()
        .iter()
        .map(|read| {
            let [n, k] = read.local_shape();
            NumericFp8LinearPlan::new(
                NumericFp8Layout {
                    n,
                    k,
                    row_origin: read.weight_read().rows().start,
                    column_origin: read.weight_read().columns().start,
                    scale_type: match read.source().encoding() {
                        crate::checkpoint::NumericFp8Encoding::E4M3FnBlock128Bf16 => {
                            NumericFp8ScaleType::Bf16
                        }
                        crate::checkpoint::NumericFp8Encoding::E4M3FnBlock128F32 => {
                            NumericFp8ScaleType::F32
                        }
                    },
                },
                rows,
                scratch,
                precision.backend(),
            )
        })
        .collect()
}

impl ResidentRankLinear {
    pub fn upload(
        ops: &CudaOperators,
        prepared: &PreparedNumericFp8RankLinear,
        max_rows: usize,
        scratch: usize,
        precision: NumericFp8Precision,
    ) -> Result<Self> {
        let plan = prepared.plan();
        let plans = plans(plan, max_rows, scratch, precision)?;
        // Validate both snapshots together at preparation only. No source guard
        // escapes into the runtime execution path: the GPU image is immutable.
        plan.reads()[0].source().validate_source_identity()?;
        let mut artifacts = Vec::with_capacity(prepared.segments().len());
        for segment in prepared.segments() {
            artifacts.push(
                ops.upload_validated_numeric_fp8_linear(segment.artifact().validated_payload())?,
            );
        }
        let largest = *plans
            .iter()
            .max_by_key(|p| p.workspace_requirements().bytes)
            .unwrap();
        let workspace = ops.numeric_fp8_linear_workspace(largest)?;
        plan.reads()[0].source().validate_source_identity()?;
        Ok(Self {
            plan: plan.clone(),
            artifacts,
            workspace,
            max_rows,
            scratch_bytes: scratch,
            precision,
        })
    }

    fn preflight(
        &self,
        ops: &CudaOperators,
        input: &Rows,
        output: &[CudaF32Buffer],
    ) -> Result<Vec<NumericFp8LinearPlan>> {
        if input.shape().rows() > self.max_rows
            || input.shape().width() != self.plan.local_shape()[1]
            || output.len() != self.artifacts.len()
            || self.workspace.is_poisoned()
        {
            return Err(cuda_error(
                "numeric rank activation/segments/capacity mismatch",
            ));
        }
        let input = input.cuda()?;
        input.validate_owner(ops)?;
        let input = input
            .f32_buffer()
            .ok_or_else(|| cuda_error("numeric rank requires F32 input"))?;
        let rows = input.len() / self.plan.local_shape()[1];
        let plans = plans(
            &self.plan,
            rows,
            self.workspace.allocated_bytes(),
            self.precision,
        )?;
        let context = ops.stream_clone();
        for (output, plan) in output.iter().zip(&plans) {
            if !Arc::ptr_eq(output.as_device_buffer().context(), context.context())
                || output.len() != checked_product(rows, plan.layout().n, 1)?
                || plan.workspace_requirements().bytes as usize > self.workspace.allocated_bytes()
                || overlaps(input, output)?
            {
                return Err(cuda_error(
                    "numeric rank output owner/extent/alias/workspace mismatch",
                ));
            }
        }
        // Outputs are independently mutable kernel destinations, even when the
        // caller created them as aliases of one larger allocation.
        for i in 0..output.len() {
            for j in 0..i {
                if overlaps(&output[i], &output[j])? {
                    return Err(cuda_error("numeric rank output segments overlap"));
                }
            }
        }
        Ok(plans)
    }
}

impl CudaStandardDecoderOperators {
    /// Install one explicit rank directory before uploading rank-local numeric
    /// weights. This is a prerequisite seam, NOT an enablement of hybrid TP
    /// decoding or its collectives. Existing TP1 preparation is unchanged.
    /// `max_resident_grant_bytes` covers encoded weights and per-binding scratch;
    /// use `cuda_execution_budget` to admit caller buffers separately.
    pub fn configure_numeric_fp8_rank(
        &mut self,
        layout: Arc<HybridRankLayout>,
        max_resident_grant_bytes: usize,
    ) -> Result<()> {
        if self.numeric.is_none() || self.needs_quarantine() || self.root_submission.is_some() {
            return Err(cuda_error(
                "numeric rank configuration requires an idle numeric owner",
            ));
        }
        self.bindings
            .configure_numeric_rank(layout, max_resident_grant_bytes)
    }

    /// Upload at preparation time and return a generation-bound lookup token.
    /// Does not keep the prepared host payload in the CUDA image. Rebinding the
    /// same exact recipe is idempotent; changes of rank, role, image, capacity or
    /// arithmetic are errors, never cache misses or silent full-weight fallback.
    pub fn prepare_numeric_fp8_rank_linear(
        &mut self,
        prepared: &PreparedNumericFp8RankLinear,
        max_rows: usize,
    ) -> Result<CudaNumericFp8RankBinding> {
        if self.needs_quarantine() || self.root_submission.is_some() {
            return Err(cuda_error(
                "numeric rank preparation requires an idle owner",
            ));
        }
        let state = self
            .numeric
            .as_ref()
            .ok_or_else(|| cuda_error("numeric profile required"))?;
        self.bindings.prepare_numeric_rank(
            &self.ops,
            prepared,
            max_rows,
            state.stats.reserved_bytes,
            state.precision,
        )
    }

    pub fn numeric_fp8_rank_resident_grant_bytes(&self) -> usize {
        self.bindings
            .numeric_rank
            .as_ref()
            .map_or(0, |rank| rank.grant_bytes)
    }

    /// Execute local GPU projections into caller-preallocated contiguous
    /// [M, segment-N] outputs. For GDN these are Q/K/V in plan order, with their
    /// global scale origins unchanged. Row-parallel output is an unreduced
    /// partial. The model schedule owner is responsible for reductions/gathers.
    /// No device allocations, payload/source I/O, D2H, or CPU math occur here.
    /// Completion is fenced; on unknown completion all operand roots are held
    /// by the owner and follow its existing quarantine lifecycle.
    pub fn numeric_fp8_rank_linear_into_segments(
        &mut self,
        binding: &CudaNumericFp8RankBinding,
        input: &Rows,
        output: &mut [CudaF32Buffer],
    ) -> Result<()> {
        if self.needs_quarantine() || self.root_submission.is_some() {
            return Err(cuda_error(
                "numeric rank execution requires an idle, healthy owner",
            ));
        }
        self.numeric_fp8_rank_linear_fenced(binding, input, output)
    }

    /// Rank-owner-only seam: retire the lexical root before entering the
    /// synchronous numeric launch. The public standalone API remains idle-only.
    pub(in crate::transformer::standard::cuda) fn numeric_fp8_rank_linear_in_root(
        &mut self,
        binding: &CudaNumericFp8RankBinding,
        input: &Rows,
        output: &mut [CudaF32Buffer],
    ) -> Result<()> {
        if self.hybrid_rank.is_none() || self.needs_quarantine() {
            return Err(cuda_error(
                "numeric root execution requires a healthy rank owner",
            ));
        }
        self.root_barrier(super::super::BarrierReason::OpaqueOperator)?;
        self.numeric_fp8_rank_linear_fenced(binding, input, output)
    }

    fn numeric_fp8_rank_linear_fenced(
        &mut self,
        binding: &CudaNumericFp8RankBinding,
        input: &Rows,
        output: &mut [CudaF32Buffer],
    ) -> Result<()> {
        self.f32(input)?;
        let plans = self
            .bindings
            .numeric_rank_linear(binding)?
            .preflight(&self.ops, input, output)?;
        self.consumer_proof()?;
        let hold_start = self.expert_hold.len();
        self.expert_hold.push(
            self.f32(input)?
                .as_device_buffer()
                .slice(0, input.shape().elements())?,
        );
        for buffer in output.iter() {
            self.expert_hold
                .push(buffer.as_device_buffer().slice(0, buffer.len())?);
        }
        let ops = Rc::clone(&self.ops);
        let resident = self.bindings.numeric_rank_linear_mut(binding)?;
        let input = input.cuda()?.f32_buffer().expect("preflighted F32 input");
        let result = resident
            .artifacts
            .iter()
            .zip(output)
            .zip(plans)
            .try_for_each(|((weight, output), plan)| {
                ops.numeric_fp8_linear_into(
                    weight,
                    input,
                    output,
                    &mut resident.workspace,
                    plan,
                    plan.layout().k,
                    plan.layout().n,
                )
            });
        if resident.workspace.is_poisoned() {
            self.poisoned = true;
            self.cache_quarantine();
        }
        let proof = self.consumer_proof();
        if proof.is_ok() {
            self.expert_hold.truncate(hold_start);
        }
        result.and(proof)
    }
}

impl ResidentRankLinear {
    pub fn matches_execution(
        &self,
        rows: usize,
        scratch: usize,
        precision: NumericFp8Precision,
    ) -> bool {
        self.max_rows == rows && self.scratch_bytes == scratch && self.precision == precision
    }
}

fn checked_product(a: usize, b: usize, c: usize) -> Result<usize> {
    a.checked_mul(b)
        .and_then(|n| n.checked_mul(c))
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| cuda_error("numeric rank extent overflow"))
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| cuda_error("numeric rank grant overflow"))
}
fn grant(n: usize) -> Result<usize> {
    ferrule_backend::cuda_allocation_grant_bytes(n)
        .ok_or_else(|| cuda_error("numeric rank allocation class overflow"))
}
fn overlaps(a: &CudaF32Buffer, b: &CudaF32Buffer) -> Result<bool> {
    let start_a = a.as_device_buffer().cu_deviceptr();
    let start_b = b.as_device_buffer().cu_deviceptr();
    let end_a = start_a
        .checked_add(checked_product(a.len(), 4, 1)? as u64)
        .ok_or_else(|| cuda_error("numeric rank device extent overflow"))?;
    let end_b = start_b
        .checked_add(checked_product(b.len(), 4, 1)? as u64)
        .ok_or_else(|| cuda_error("numeric rank device extent overflow"))?;
    Ok(start_a < end_b && start_b < end_a)
}
