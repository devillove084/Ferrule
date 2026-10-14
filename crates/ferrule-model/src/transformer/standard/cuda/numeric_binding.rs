//! Explicit numeric FP8 storage profile and one fenced, reusable workspace.
#[path = "numeric_shard.rs"]
pub(super) mod rank;
use std::rc::Rc;

use ferrule_backend::cuda::operators::linear::{
    CudaF32Buffer, CudaNumericFp8Artifact, CudaNumericFp8Workspace, CudaOperators,
    NumericFp8LinearPlan,
};
use ferrule_common::Result;

use super::{CudaStandardDecoderOperators, ExpertCachePolicy, cuda_error};
use crate::checkpoint::NumericFp8Artifact;
use crate::transformer::{BoundParameter, NumericFp8Precision, PreparedLinear, Rows};

pub(super) struct NumericBinding {
    pub artifact: CudaNumericFp8Artifact,
    pub source: crate::checkpoint::NumericFp8Source,
    pub role: crate::support::TensorRole,
}

pub(super) fn upload(
    ops: &CudaOperators,
    artifact: &NumericFp8Artifact,
) -> Result<CudaNumericFp8Artifact> {
    artifact.provenance().source().validate_source_identity()?;
    let payload = artifact.validated_payload();
    #[cfg(test)]
    tests::PROOF_UPLOADS.with(|calls| {
        calls.borrow_mut().push((
            payload.weight_bytes().as_ptr() as usize,
            payload.scale_bytes().as_ptr() as usize,
        ));
    });
    ops.upload_validated_numeric_fp8_linear(payload)
}

pub(super) fn validate_linear(linear: &PreparedLinear) -> Result<()> {
    let artifact = linear
        .numeric_fp8()
        .ok_or_else(|| cuda_error("missing numeric artifact"))?;
    let source = artifact.provenance().source();
    source.validate_source_identity()?;
    let [rows, columns] = linear.global_shape();
    let read = artifact.provenance().weight_read();
    if artifact.local_shape() != [rows, columns]
        || read.rows().start != 0
        || read.columns().start != 0
        || source.expert_count().is_some()
        || source.weight() != linear.parameter().binding().weight().slice()
        || linear.parameter().binding().scale().map(|p| p.slice()) != Some(source.scale())
        || linear.role() != linear.parameter().role()
        || linear.tensor_shard().is_some()
    {
        return Err(cuda_error(
            "numeric linear shape/role/paired provenance mismatch",
        ));
    }
    Ok(())
}

/// Owner-local numeric scratch, included in `ExpertCacheStats::scratch_bytes`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NumericWorkspaceStats {
    /// Charged for the lifetime of the numeric owner, including between calls.
    pub reserved_bytes: usize,
    /// Live reusable workspace payload, never greater than `reserved_bytes`.
    pub allocated_bytes: usize,
    /// Workspace handle creations, including a zero-payload fused workspace.
    /// Consult `allocated_bytes` for live global scratch, not this counter.
    pub allocations: u64,
    pub reuses: u64,
    pub submissions: u64,
}

#[cfg(test)]
type ScopeStepHook = (usize, fn(&mut CudaStandardDecoderOperators) -> Result<()>);

pub(super) struct NumericState {
    pub precision: NumericFp8Precision,
    pub workspace: Option<CudaNumericFp8Workspace>,
    pub stats: NumericWorkspaceStats,
    pub operation_bytes: usize,
    #[cfg(test)]
    scope_step_hook: Option<ScopeStepHook>,
}

impl CudaStandardDecoderOperators {
    /// Explicit numeric FP8 profile. Native F32/BF16 projections retain the
    /// legacy TF32x3 path. Numeric projections use BF16-RNE operands and F32
    /// accumulation/output. The scratch reservation is charged to bounded
    /// expert residency; it is not an additional hidden allocation budget.
    ///
    /// For the 35B runner use `Bounded(ExpertCacheLimits { max_experts: 64,
    /// max_bytes: 320 * 1024 * 1024 })` and `64 * 1024 * 1024` scratch bytes.
    /// Non-expert weights, KV and allocator segment slack need separate budgets.
    pub fn new_numeric_fp8(
        ops: Rc<CudaOperators>,
        parameters: &[BoundParameter],
        policy: ExpertCachePolicy,
        numeric_scratch_bytes: usize,
    ) -> Result<Self> {
        Self::new_numeric_fp8_with_precision(
            ops,
            parameters,
            policy,
            numeric_scratch_bytes,
            NumericFp8Precision::Bf16RneF32Accumulate,
        )
    }

    /// Creates an image with immutable numeric arithmetic. Resident weights
    /// remain compressed in every mode. Scratch (including F32 decoded tiles)
    /// is bounded by the same reservation and reused only after consumer proof.
    /// W8A16 permits a zero reservation: it rounds operands in on-chip tiles
    /// without global numeric scratch. Operation buffers are still charged to
    /// the expert budget; a nonzero numeric reservation is not silently waived.
    pub fn new_numeric_fp8_with_precision(
        ops: Rc<CudaOperators>,
        parameters: &[BoundParameter],
        policy: ExpertCachePolicy,
        numeric_scratch_bytes: usize,
        precision: NumericFp8Precision,
    ) -> Result<Self> {
        if precision.has_bf16_model_boundaries() && !matches!(policy, ExpertCachePolicy::Bounded(_))
        {
            return Err(cuda_error(
                "numeric BF16 model requires bounded expert residency",
            ));
        }
        if numeric_scratch_bytes == 0 && !precision.allows_zero_scratch() {
            return Err(cuda_error(
                "numeric scratch budget must be nonzero unless W8A16",
            ));
        }
        let mut owner = Self::new_with_expert_profile(
            ops,
            crate::execution::ExecutionPrecisionPolicy::f32(),
            parameters,
            policy,
            true,
        )?;
        if precision.has_bf16_model_boundaries() {
            owner.bf16 = Some(super::bf16::State::default());
            owner.bindings.bf16 = true;
        }
        owner.numeric = Some(NumericState {
            precision,
            workspace: None,
            operation_bytes: 0,
            #[cfg(test)]
            scope_step_hook: None,
            stats: NumericWorkspaceStats {
                reserved_bytes: numeric_scratch_bytes,
                ..Default::default()
            },
        });
        owner.cache_scratch(0)?;
        Ok(owner)
    }

    pub fn numeric_fp8_precision(&self) -> Option<NumericFp8Precision> {
        self.numeric.as_ref().map(|state| state.precision)
    }

    pub fn numeric_workspace_stats(&self) -> Option<super::NumericWorkspaceStats> {
        self.numeric.as_ref().map(|state| state.stats)
    }

    pub(super) fn numeric_linear_into(
        &mut self,
        binding: &NumericBinding,
        input: &Rows,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        binding.source.validate_source_identity()?;
        if self.bf16.is_some() {
            self.bf16_buffer(input)?;
        } else {
            self.f32(input)?;
        }
        let layout = binding.artifact.layout();
        if input.shape().width() != layout.k
            || input.shape().rows().checked_mul(layout.n) != Some(output.len())
        {
            return Err(cuda_error("numeric activation/output layout mismatch"));
        }
        let state = self
            .numeric
            .as_ref()
            .ok_or_else(|| cuda_error("numeric FP8 profile required"))?;
        let plan = state.precision.linear_plan(
            &binding.role,
            binding.artifact.layout(),
            input.shape().rows(),
            state.stats.reserved_bytes,
        )?;
        // Only a semantic operation with owner-pinned operands may coalesce
        // these same-stream workspace reuse proofs. Growth is never deferred.
        if !self.numeric_workspace_needs_growth(plan) {
            self.ordered_consumer_proof()?;
        }
        self.reserve_numeric_workspace(plan)?;
        self.submit_numeric_linear(binding, input, output, plan)?;
        self.ordered_consumer_proof()
    }

    fn numeric_workspace_needs_growth(&self, plan: NumericFp8LinearPlan) -> bool {
        self.numeric
            .as_ref()
            .and_then(|state| state.workspace.as_ref())
            .is_none_or(|workspace| {
                workspace.allocated_bytes() < plan.workspace_requirements().bytes as usize
                    || workspace.precision() != plan.precision()
            })
    }

    /// Growth or precision rebinding destroys storage, unlike same-stream reuse.
    /// Keep one bounded workspace live. Always fence before
    /// replacing it, even when called from a preflighted SwiGLU scope.
    fn reserve_numeric_workspace(&mut self, plan: NumericFp8LinearPlan) -> Result<()> {
        if self.needs_quarantine() {
            return Err(cuda_error("numeric workspace completion remains unknown"));
        }
        let required = plan.workspace_requirements().bytes as usize;
        if self.numeric_workspace_needs_growth(plan) {
            // A growth/rebind is a real storage retirement boundary. The
            // caller may not suppress this fence even inside a scope.
            if self.root_submission.is_some() {
                self.root_barrier(super::BarrierReason::WorkspaceGrowth)?;
            } else {
                self.consumer_proof()?;
            }
            let state = self.numeric.as_mut().expect("numeric profile");
            state.workspace = None;
            state.stats.allocated_bytes = 0;
            // The image admission adds this persistent reserve once, independently
            // of the bounded cache/scratch grant. Initialization precedes submission.
            self.ops
                .prepare_numeric_fp8_provider(plan, plan.provider_bytes())?;
            let workspace = self.ops.numeric_fp8_linear_workspace(plan)?;
            let state = self.numeric.as_mut().expect("numeric profile");
            state.workspace = Some(workspace);
            state.stats.allocated_bytes = required;
            state.stats.allocations += 1;
        } else {
            self.numeric.as_mut().expect("numeric profile").stats.reuses += 1;
        }
        Ok(())
    }

    // Only standalone linear (with its original fences) and the preallocated,
    // owner-pinned SwiGLU scope may reach this submission-only seam.
    fn submit_numeric_linear(
        &mut self,
        binding: &NumericBinding,
        input: &Rows,
        output: &mut CudaF32Buffer,
        plan: NumericFp8LinearPlan,
    ) -> Result<()> {
        if self.needs_quarantine() {
            return Err(cuda_error("numeric submission completion remains unknown"));
        }
        if self.bf16.is_some() {
            self.bf16_buffer(input)?;
        } else {
            self.f32(input)?;
        }
        let rows = input.cuda()?;
        let state = self.numeric.as_mut().expect("numeric profile");
        let workspace = state.workspace.as_mut().expect("numeric workspace");
        let result = if state.precision.has_bf16_model_boundaries() {
            self.ops.numeric_fp8_bf16_linear_into(
                &binding.artifact,
                rows.bf16_buffer()
                    .ok_or_else(|| cuda_error("numeric BF16 activation required"))?,
                output,
                workspace,
                plan,
                binding.artifact.layout().k,
                binding.artifact.layout().n,
            )
        } else {
            self.ops.numeric_fp8_linear_into(
                &binding.artifact,
                rows.f32_buffer()
                    .ok_or_else(|| cuda_error("numeric F32 activation required"))?,
                output,
                workspace,
                plan,
                binding.artifact.layout().k,
                binding.artifact.layout().n,
            )
        };
        state.stats.submissions += 1;
        if workspace.is_poisoned() {
            self.poisoned = true;
            self.cache_quarantine();
        }
        result
    }

    /// The root's synchronous dense_swiglu call is the lexical guard: no rows
    /// escape before its final proof. Only already-resident layer parameters
    /// qualify; misses/uploads, observers (which may read on the CPU), TP and
    /// expert leases retain their original paths and fences.
    pub(super) fn root_numeric_swiglu(
        &mut self,
        expert: &crate::transformer::PreparedSwiGlu,
        input: &Rows,
        arena: Option<crate::transformer::RowsArenaId>,
    ) -> Result<Option<Rows>> {
        let projections = [expert.gate(), expert.up(), expert.down()];
        let residency = expert.gate().parameter().residency();
        if self.numeric.is_none()
            || self.bindings.tensor.is_some()
            || self.bindings.expert_admission.is_some()
            || self.diagnostic.is_enabled()
            || !matches!(residency, crate::nn::ParameterResidency::Layer { .. })
            || projections.iter().any(|p| {
                p.numeric_fp8().is_none()
                    || p.bias().is_some()
                    || p.tensor_shard().is_some()
                    || p.parameter().binding().is_alias()
                    || p.parameter().residency() != residency
            })
            // Do not combine retained operations under a single-op budget.
            || !self.expert_hold.is_empty()
            || !self.expert_artifacts.is_empty()
            || !self.route_hold.is_empty()
            || !self.route_ids.is_empty()
            || self.route_scratch_bytes != 0
        {
            return Ok(None);
        }
        for projection in projections {
            self.bindings.validate(projection.parameter())?;
            validate_linear(projection)?;
        }
        // Same live input + three intermediates + output as swiglu_compute:
        // its existing owner holds already retain all five until this fence.
        // Artifacts are resident image roots, not additional device allocations.
        let bytes =
            crate::transformer::SwiGluScratchPlan::for_prepared(expert, input.shape().rows())?
                .total_bytes()?;
        self.reserve_numeric_temporaries(bytes)?;
        let parameters = projections.map(|p| p.parameter().binding().clone());
        self.resident_numeric_swiglu_inner(
            &parameters,
            Some(expert),
            expert.activation_limit(),
            input,
            arena,
        )
    }

    /// A lexical, owner-local compute scope, not an asynchronous public API.
    /// The caller's admission/synchronous boundary owns the final consumer proof.
    /// A miss, mixed storage, bias or TP keeps the ordinary path.
    pub(super) fn resident_numeric_swiglu(
        &mut self,
        parameters: &[BoundParameter; 3],
        prepared: Option<&crate::transformer::PreparedSwiGlu>,
        activation_limit: Option<f32>,
        input: &Rows,
        arena: Option<crate::transformer::RowsArenaId>,
    ) -> Result<Option<Rows>> {
        if self.numeric.is_none()
            || self.bindings.tensor.is_some()
            || self.bindings.expert_admission.is_none()
        {
            return Ok(None);
        }
        self.resident_numeric_swiglu_inner(parameters, prepared, activation_limit, input, arena)
    }

    fn resident_numeric_swiglu_inner(
        &mut self,
        parameters: &[BoundParameter; 3],
        prepared: Option<&crate::transformer::PreparedSwiGlu>,
        activation_limit: Option<f32>,
        input: &Rows,
        arena: Option<crate::transformer::RowsArenaId>,
    ) -> Result<Option<Rows>> {
        use super::binding::ResidentLinearStorage;
        use crate::transformer::RowsShape;
        let mut weights = Vec::with_capacity(3);
        for parameter in parameters {
            let Some(weight) = self.bindings.metadata_linear(parameter)? else {
                return Ok(None);
            };
            if !matches!(weight.storage, ResidentLinearStorage::Numeric(_)) {
                return Ok(None);
            }
            weights.push(weight);
        }
        self.f32(input)?;
        let rows = input.shape().rows();
        let state = self.numeric.as_ref().expect("numeric profile");
        let plans = weights
            .iter()
            .map(|weight| {
                let ResidentLinearStorage::Numeric(binding) = &weight.storage else {
                    unreachable!()
                };
                binding.source.validate_source_identity()?;
                NumericFp8LinearPlan::new(
                    binding.artifact.layout(),
                    rows,
                    state.stats.reserved_bytes,
                    state.precision.backend(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        if input.shape().width() != weights[0].shape.1
            || weights[0].shape != weights[1].shape
            || weights[2].shape.1 != weights[0].shape.0
            || activation_limit.is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            return Err(cuda_error("numeric SwiGLU layout/limit mismatch"));
        }
        let hidden = RowsShape::new(rows, weights[0].shape.0)?;
        let output_shape = RowsShape::new(rows, weights[2].shape.0)?;
        // All three artifact roots live on the owner BEFORE any allocation or
        // launch can fail/unwind. Either the expert lease pins their credit or
        // the root image already owns them for its entire lifetime.
        self.expert_artifacts.extend(weights.iter().cloned());
        self.expert_hold.push(
            self.f32(input)?
                .as_device_buffer()
                .slice(0, input.shape().elements())?,
        );
        let peak = *plans
            .iter()
            .max_by_key(|p| p.workspace_requirements().bytes)
            .expect("three plans");
        self.reserve_numeric_workspace(peak)?;
        let mut allocate = |_index, elements| -> Result<CudaF32Buffer> {
            #[cfg(test)]
            if self.numeric_allocation_failure == Some(_index) {
                self.numeric_allocation_failure = None;
                self.ops.failpoints().arm_allocation();
            }
            let buffer = self.ops.zero_f32_buffer(elements)?;
            self.hold_expert_buffer(&buffer)?;
            Ok(buffer)
        };
        // No allocation, eviction, workspace replacement or rebind between
        // gate/up/activation/down. Same compute-stream order protects scratch.
        let mut gate = allocate(0, hidden.elements())?;
        let mut up = allocate(1, hidden.elements())?;
        let mut product = allocate(2, hidden.elements())?;
        let mut output = allocate(3, output_shape.elements())?;
        let project = |this: &mut Self,
                       index: usize,
                       input: &Rows,
                       output: &mut CudaF32Buffer|
         -> Result<()> {
            let ResidentLinearStorage::Numeric(binding) = &weights[index].storage else {
                unreachable!()
            };
            let trace = |this: &Self, name, buffer: &CudaF32Buffer, shape| {
                if let Some(expert) = prepared {
                    let projection = [expert.gate(), expert.up(), expert.down()][index];
                    this.trace_buffer(name, buffer, shape, Some(projection), None, None)
                } else {
                    this.trace_expert_linear(name, buffer, shape, &parameters[index])
                }
            };
            trace(this, "linear.input", this.f32(input)?, input.shape())?;
            this.submit_numeric_linear(binding, input, output, plans[index])?;
            let shape = RowsShape::new(rows, weights[index].shape.0)?;
            trace(this, "linear.output", output, shape)
        };
        project(self, 0, input, &mut gate)?;
        #[cfg(test)]
        self.numeric_scope_step(0)?;
        project(self, 1, input, &mut up)?;
        #[cfg(test)]
        self.numeric_scope_step(1)?;
        match activation_limit {
            Some(limit) => self
                .ops
                .swiglu_f32_clamped_into(&gate, &up, &mut product, limit)?,
            None => self.ops.swiglu_f32_into(&gate, &up, &mut product)?,
        }
        #[cfg(test)]
        self.numeric_scope_step(2)?;
        let product = super::device_rows(hidden, arena, product)?;
        project(self, 2, &product, &mut output)?;
        #[cfg(test)]
        self.numeric_scope_step(3)?;
        let output = super::device_rows(output_shape, arena, output)?;
        self.trace_rows("swiglu.output", &output)?;
        Ok(Some(output))
    }

    #[cfg(test)]
    fn numeric_scope_step(&mut self, step: usize) -> Result<()> {
        if let Some((target, hook)) = self.numeric.as_ref().unwrap().scope_step_hook
            && target == step
        {
            hook(self)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{CheckpointDType, CheckpointTensorSlice};
    use crate::nn::{ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec};
    use crate::support::TensorRole;
    use crate::transformer::{
        ExactNameMapper, ExpertCacheLimits, HostRows, NameMapping, PreparedSwiGlu, RowsDType,
        RowsShape, StandardDecoderOperators, StateDictBinder, StateDictMaterializer,
        StateDictSchema,
    };

    mod root_scope {
        use super::*;
        use crate::transformer::{
            MoeRouterSpec, OperatorProgress, RouterScoreFunction, RouterSelection,
        };

        mod submission {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/unit/transformer/root_execution_cuda.rs"
            ));
        }

        const RESERVATION: usize = 4096;

        struct Fixture {
            path: std::path::PathBuf,
            parameters: Vec<BoundParameter>,
            expert: PreparedSwiGlu,
        }

        impl Fixture {
            fn new() -> Self {
                use std::sync::atomic::{AtomicU64, Ordering};
                static NEXT: AtomicU64 = AtomicU64::new(0);
                let path = std::env::temp_dir().join(format!(
                    "root-numeric-scope-{}-{}.bin",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                let mut payload = vec![0x38; 96];
                payload.extend(0x3d80u16.to_le_bytes());
                std::fs::write(&path, payload).unwrap();
                let mut schema = StateDictSchema::builder();
                let mut mapper = ExactNameMapper::new();
                let mut slices = Vec::new();
                for (index, role) in [
                    TensorRole::SharedExpertGate,
                    TensorRole::SharedExpertUp,
                    TensorRole::SharedExpertDown,
                ]
                .into_iter()
                .enumerate()
                {
                    let name = format!("shared.p{index}");
                    let module = ModulePath::new(&name).unwrap();
                    let shape = if index == 2 { vec![8, 12] } else { vec![12, 8] };
                    schema
                        .register_with_role(
                            ParameterSpec::new(
                                ParameterId::new(index as u64 + 1),
                                module.clone(),
                                ParameterDType::F8E4M3,
                                shape.clone(),
                                ParameterResidency::layer(0),
                            )
                            .unwrap()
                            .with_required_scale(ParameterDType::Bf16, [1, 1])
                            .unwrap(),
                            role.clone(),
                        )
                        .unwrap();
                    mapper
                        .insert(&name, NameMapping::weight(module.clone()))
                        .unwrap();
                    mapper
                        .insert(format!("{name}_scale"), NameMapping::scale(module))
                        .unwrap();
                    slices.push(CheckpointTensorSlice {
                        name: name.clone(),
                        path: path.clone(),
                        role: role.clone(),
                        offset: 0,
                        bytes: 96,
                        shape,
                        dtype: CheckpointDType::F8E4M3,
                    });
                    slices.push(CheckpointTensorSlice {
                        name: format!("{name}_scale"),
                        path: path.clone(),
                        role,
                        offset: 96,
                        bytes: 2,
                        shape: vec![1, 1],
                        dtype: CheckpointDType::Bf16,
                    });
                }
                let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
                    .bind_slices(slices)
                    .unwrap();
                let materializer = StateDictMaterializer::new(1024).unwrap();
                let linear = |id| {
                    let p = bound.get_by_id(ParameterId::new(id)).unwrap();
                    materializer.prepared_linear(p, p.role().clone()).unwrap()
                };
                let expert =
                    PreparedSwiGlu::new(linear(1), linear(2), linear(3), Some(0.5)).unwrap();
                Self {
                    path,
                    parameters: bound.parameters().to_vec(),
                    expert,
                }
            }

            fn owner(
                &self,
                precision: NumericFp8Precision,
                cap: usize,
            ) -> CudaStandardDecoderOperators {
                CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
                    Rc::new(CudaOperators::new_on_device(0).unwrap()),
                    &self.parameters,
                    ExpertCachePolicy::Bounded(ExpertCacheLimits {
                        max_experts: 1,
                        max_bytes: cap,
                    }),
                    RESERVATION,
                    precision,
                )
                .unwrap()
            }
        }

        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.path);
            }
        }

        fn precisions() -> [NumericFp8Precision; 2] {
            [
                NumericFp8Precision::F32Tf32x3,
                NumericFp8Precision::Bf16RneF32Accumulate,
            ]
        }

        fn input(owner: &mut CudaStandardDecoderOperators, rows: usize) -> Rows {
            owner
                .bind_rows(Rows::Host(
                    HostRows::new(
                        RowsShape::new(rows, 8).unwrap(),
                        RowsDType::F32,
                        None,
                        (0..rows * 8)
                            .map(|i| (i % 11) as f32 * 0.031 - 0.1)
                            .collect(),
                    )
                    .unwrap(),
                ))
                .unwrap()
        }

        fn ready(progress: OperatorProgress<Rows>) -> Rows {
            let OperatorProgress::Ready(rows) = progress else {
                panic!("expected ready")
            };
            rows
        }

        fn warm(owner: &mut CudaStandardDecoderOperators, expert: &PreparedSwiGlu, input: &Rows) {
            drop(ready(owner.dense_swiglu(expert, input, None).unwrap()));
        }

        fn assert_roots(owner: &CudaStandardDecoderOperators) {
            assert_eq!(owner.expert_artifacts.len(), 3);
            assert_eq!(owner.expert_hold.len(), 5);
            assert!(owner.bindings.expert_admission.is_none());
            assert!(
                owner
                    .expert_artifacts
                    .iter()
                    .all(|w| Rc::strong_count(w) >= 2)
            );
            assert!(owner.numeric.as_ref().unwrap().operation_bytes > 0);
        }

        #[test]
        #[ignore = "requires CUDA; real root proof counts, same arithmetic and no additional peak"]
        fn root_scope_reduces_seven_proofs_to_one_at_the_same_peak() {
            for precision in precisions() {
                for rows in [1, 3, 16] {
                    let fixture = Fixture::new();
                    let mut owner = fixture.owner(precision, 32768);
                    let input = input(&mut owner, rows);
                    warm(&mut owner, &fixture.expert, &input);
                    let workspace = owner.numeric_workspace_stats().unwrap();
                    let before = owner.consumer_proofs;
                    owner.ops.reset_counters();
                    let expected = owner
                        .synchronous(|this| this.swiglu_rows(&fixture.expert, &input, None))
                        .unwrap();
                    assert_eq!(owner.consumer_proofs - before, 7);
                    let allocations = owner.ops.counters().device_allocation_attempts;
                    let peak = owner.ops.allocator_metrics().peak_live_requested_bytes;
                    let expected = owner.download_rows(expected).unwrap();
                    let before = owner.consumer_proofs;
                    owner.ops.reset_counters();
                    let actual = ready(owner.dense_swiglu(&fixture.expert, &input, None).unwrap());
                    assert_eq!(owner.consumer_proofs - before, 1);
                    assert_eq!(owner.ops.counters().device_allocation_attempts, allocations);
                    assert_eq!(
                        owner.ops.allocator_metrics().peak_live_requested_bytes,
                        peak
                    );
                    assert_eq!(
                        owner.numeric_workspace_stats().unwrap().allocations,
                        workspace.allocations
                    );
                    assert!(owner.expert_hold.is_empty());
                    assert!(owner.expert_artifacts.is_empty());
                    let actual = owner.download_rows(actual).unwrap();
                    assert_eq!(actual.values(), expected.values());
                    assert_eq!(
                        owner.expert_cache_stats().unwrap().scratch_bytes,
                        RESERVATION
                    );
                }
            }
        }

        #[test]
        #[ignore = "requires CUDA; cold miss, foreign owner and parameter rebind never enter the root scope"]
        fn root_scope_cold_and_rebind_fallback_precedes_submission() {
            for precision in precisions() {
                let fixture = Fixture::new();
                let mut owner = fixture.owner(precision, 32768);
                let input = input(&mut owner, 3);
                let before = owner.consumer_proofs;
                warm(&mut owner, &fixture.expert, &input);
                assert_eq!(
                    owner.consumer_proofs - before,
                    7,
                    "cold path keeps projection proofs"
                );
                assert_eq!(owner.resident_parameter_bytes(), 294);
                let before = owner.consumer_proofs;
                warm(&mut owner, &fixture.expert, &input);
                assert_eq!(owner.consumer_proofs - before, 1);

                let mut other = fixture.owner(precision, 32768);
                other.prepare_expert(&fixture.expert).unwrap();
                assert!(other.dense_swiglu(&fixture.expert, &input, None).is_err());
                assert_eq!(other.numeric_workspace_stats().unwrap().submissions, 0);
                assert!(!other.needs_quarantine());

                let before = owner.numeric_workspace_stats().unwrap().submissions;
                let replacement = fixture.path.with_extension("replacement");
                std::fs::copy(&fixture.path, &replacement).unwrap();
                std::fs::rename(&replacement, &fixture.path).unwrap();
                assert!(owner.dense_swiglu(&fixture.expert, &input, None).is_err());
                assert_eq!(owner.numeric_workspace_stats().unwrap().submissions, before);
                assert!(owner.expert_hold.is_empty());
                assert!(owner.expert_artifacts.is_empty());
                assert!(!owner.needs_quarantine());
            }
        }

        #[test]
        #[ignore = "requires CUDA; root scope must end before standalone/CPU/router/transfer boundaries"]
        fn root_scope_preserves_standalone_and_host_boundary_proofs() {
            let fixture = Fixture::new();
            let mut owner = fixture.owner(NumericFp8Precision::F32Tf32x3, 32768);
            let input = input(&mut owner, 1);
            warm(&mut owner, &fixture.expert, &input);
            let before = owner.consumer_proofs;
            drop(ready(
                owner.linear(fixture.expert.gate(), &input, None).unwrap(),
            ));
            assert_eq!(owner.consumer_proofs - before, 1); // resident projection scope
            let output = ready(owner.dense_swiglu(&fixture.expert, &input, None).unwrap());
            let policy = MoeRouterSpec::new(
                8,
                2,
                RouterScoreFunction::Softmax,
                RouterSelection::TopK,
                true,
                1.0,
            )
            .unwrap();
            let before = owner.consumer_proofs;
            assert!(matches!(
                owner.router(&output, &policy).unwrap(),
                OperatorProgress::Ready(_)
            ));
            assert_eq!(owner.consumer_proofs - before, 1);
            let before = owner.consumer_proofs;
            let host = owner.download_rows(output).unwrap();
            assert_eq!(owner.consumer_proofs - before, 1);
            let before = owner.consumer_proofs;
            drop(owner.bind_rows(Rows::Host(host)).unwrap());
            assert_eq!(owner.consumer_proofs - before, 1);
            let before = owner.consumer_proofs;
            owner.quiesce().unwrap();
            assert_eq!(owner.consumer_proofs - before, 1);

            // A diagnostic observer may perform a CPU read. It must see the
            // original individually proven projections, not the GPU-only scope.
            owner.set_diagnostic_trace(|event| {
                event.operators.download_f32_buffer(event.buffer)?;
                Ok(())
            });
            let before = owner.consumer_proofs;
            warm(&mut owner, &fixture.expert, &input);
            assert_eq!(owner.consumer_proofs - before, 7);
            owner.set_diagnostic_trace(|_| Err(cuda_error("observer cancellation")));
            assert!(owner.dense_swiglu(&fixture.expert, &input, None).is_err());
            assert!(!owner.needs_quarantine());
            owner.clear_diagnostic_trace();
            warm(&mut owner, &fixture.expert, &input);
        }

        fn fail(owner: &mut CudaStandardDecoderOperators) -> Result<()> {
            assert_roots(owner);
            Err(cuda_error("root operation cancelled"))
        }
        fn panic_after_submit(owner: &mut CudaStandardDecoderOperators) -> Result<()> {
            assert_roots(owner);
            panic!("root operation panic");
        }
        fn unknown(owner: &mut CudaStandardDecoderOperators) -> Result<()> {
            assert_roots(owner);
            owner.lose_next_consumer_proof = true;
            Err(cuda_error(
                "root operation failed before unknown completion",
            ))
        }
        fn final_unknown(owner: &mut CudaStandardDecoderOperators) -> Result<()> {
            assert_roots(owner);
            owner.lose_next_consumer_proof = true;
            Ok(())
        }
        #[test]
        #[ignore = "requires CUDA; Boundary poison cannot be washed by the root's final proof"]
        fn root_scope_boundary_unknown_is_permanent() {
            let fixture = Fixture::new();
            let mut owner = fixture.owner(NumericFp8Precision::F32Tf32x3, 32768);
            let input = input(&mut owner, 1);
            warm(&mut owner, &fixture.expert, &input);
            let result = owner.synchronous(|this| {
                let output = this
                    .root_numeric_swiglu(&fixture.expert, &input, None)?
                    .unwrap();
                assert_roots(this);
                this.boundary.fail_next_producer_record =
                    Some(cuda_error("root producer record lost"));
                this.boundary.download(&this.ops, this.f32(&output)?)
            });
            assert!(result.is_err());
            assert!(owner.needs_quarantine());
            assert_roots(&owner);
            let frozen = owner.expert_cache_stats().unwrap();
            owner.ops.sync_stream().unwrap();
            owner.ops.sync_upload_stream().unwrap();
            assert!(owner.quiesce().is_err());
            assert!(owner.finish_cache_operation().is_err());
            assert_eq!(owner.expert_cache_stats().unwrap(), frozen);
        }

        #[test]
        #[ignore = "requires CUDA; every submitted op fail/panic/cancel/Unknown and final consumer custody"]
        fn root_scope_each_operation_failure_and_unknown_retains_roots() {
            let hooks: [fn(&mut CudaStandardDecoderOperators) -> Result<()>; 4] =
                [fail, panic_after_submit, unknown, final_unknown];
            for precision in precisions() {
                for step in 0..4 {
                    for (mode, hook) in hooks.into_iter().enumerate() {
                        let fixture = Fixture::new();
                        let mut owner = fixture.owner(precision, 32768);
                        let input = input(&mut owner, 1);
                        warm(&mut owner, &fixture.expert, &input);
                        owner.numeric.as_mut().unwrap().scope_step_hook = Some((step, hook));
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            owner.dense_swiglu(&fixture.expert, &input, None)
                        }));
                        assert!(!matches!(result, Ok(Ok(_))));
                        if mode == 0 {
                            assert!(!owner.needs_quarantine());
                            assert!(owner.expert_hold.is_empty());
                            assert!(owner.expert_artifacts.is_empty());
                            owner.numeric.as_mut().unwrap().scope_step_hook = None;
                            warm(&mut owner, &fixture.expert, &input);
                        } else {
                            assert_roots(&owner);
                            assert!(owner.needs_quarantine());
                            let stats = owner.expert_cache_stats().unwrap();
                            let workspace = owner.numeric_workspace_stats().unwrap();
                            let live = owner.ops.allocator_metrics().live_requested_bytes;
                            assert_eq!(stats.unknown_quarantine_bytes, stats.charged_bytes());
                            owner.ops.sync_stream().unwrap();
                            owner.ops.sync_upload_stream().unwrap();
                            assert!(owner.quiesce().is_err());
                            assert!(owner.finish_cache_operation().is_err());
                            assert_eq!(owner.expert_cache_stats().unwrap(), stats);
                            assert_eq!(owner.numeric_workspace_stats().unwrap(), workspace);
                            let ops = Rc::clone(&owner.ops);
                            drop(input);
                            drop(owner);
                            ops.trim_device_allocator().unwrap();
                            assert!(ops.allocator_metrics().live_requested_bytes >= live);
                        }
                    }
                }
            }
        }

        #[test]
        #[ignore = "requires CUDA; exact admitted cap, growth fence and partial allocation custody"]
        fn root_scope_budget_growth_and_allocation_failures_keep_original_fences() {
            for precision in precisions() {
                let fixture = Fixture::new();
                let bytes = fixture.expert.f32_scratch_bytes(16).unwrap();
                for (cap, admitted) in [
                    (RESERVATION + bytes, true),
                    (RESERVATION + bytes - 1, false),
                ] {
                    let mut owner = fixture.owner(precision, cap);
                    let input = input(&mut owner, 16);
                    owner.prepare_expert(&fixture.expert).unwrap();
                    owner.ops.reset_counters();
                    assert_eq!(
                        owner.dense_swiglu(&fixture.expert, &input, None).is_ok(),
                        admitted
                    );
                    if !admitted {
                        assert_eq!(owner.ops.counters().device_allocation_attempts, 0);
                        assert_eq!(owner.numeric_workspace_stats().unwrap().submissions, 0);
                    }
                    assert!(owner.expert_cache_stats().unwrap().peak_bytes <= cap);
                    assert!(!owner.needs_quarantine());
                }
                for allocation in 0..4 {
                    for unknown in [false, true] {
                        let mut owner = fixture.owner(precision, 32768);
                        let input = input(&mut owner, 1);
                        warm(&mut owner, &fixture.expert, &input);
                        let submissions = owner.numeric_workspace_stats().unwrap().submissions;
                        owner.numeric_allocation_failure = Some(allocation);
                        owner.lose_next_consumer_proof = unknown;
                        assert!(owner.dense_swiglu(&fixture.expert, &input, None).is_err());
                        assert_eq!(
                            owner.numeric_workspace_stats().unwrap().submissions,
                            submissions
                        );
                        assert_eq!(owner.needs_quarantine(), unknown);
                        if unknown {
                            assert_eq!(owner.expert_artifacts.len(), 3);
                            assert_eq!(owner.expert_hold.len(), allocation + 1);
                            assert!(owner.quiesce().is_err());
                        }
                    }
                }
            }
            for unknown in [false, true] {
                let fixture = Fixture::new();
                let mut owner = fixture.owner(NumericFp8Precision::Bf16RneF32Accumulate, 32768);
                let small = input(&mut owner, 1);
                let large = input(&mut owner, 16);
                warm(&mut owner, &fixture.expert, &small);
                let before = owner.numeric_workspace_stats().unwrap();
                let proofs = owner.consumer_proofs;
                owner.lose_next_consumer_proof = unknown;
                let result = owner.dense_swiglu(&fixture.expert, &large, None);
                assert_eq!(result.is_err(), unknown);
                assert_eq!(owner.consumer_proofs - proofs, if unknown { 1 } else { 2 });
                let after = owner.numeric_workspace_stats().unwrap();
                if unknown {
                    assert_eq!(after, before);
                    assert_eq!(owner.expert_artifacts.len(), 3);
                    assert_eq!(owner.expert_hold.len(), 1);
                    assert!(owner.quiesce().is_err());
                } else {
                    assert_eq!(after.allocations, before.allocations + 1);
                    assert_eq!(after.submissions, before.submissions + 3);
                }
            }
        }
    }

    thread_local! {
        pub(super) static PROOF_UPLOADS: std::cell::RefCell<Vec<(usize, usize)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    #[test]
    #[ignore = "requires CUDA; actual prepared model binding reuses the exact CPU proof"]
    fn prepared_model_binding_uploads_proof_once_without_revalidation() {
        use crate::checkpoint::numeric_fp8::proof_tests::{Fixture, bind, counts};
        let fixture = Fixture::new(2, 2);
        let bound = bind(&fixture.source, false);
        let materializer = StateDictMaterializer::new(64).unwrap();
        let parameter = &bound.parameters()[0];
        let before = counts();
        let linear = materializer
            .prepared_linear(parameter, parameter.role().clone())
            .unwrap();
        let artifact = linear.numeric_fp8().unwrap();
        assert_eq!(counts(), (before.0 + 1, before.1 + 8));
        let addresses = (
            artifact.weight_bytes().as_ptr() as usize,
            artifact.scale_bytes().as_ptr() as usize,
        );
        let ops = CudaOperators::new_on_device(0).unwrap();
        let mut bindings = super::super::binding::Bindings::new(bound.parameters()).unwrap();
        bindings.numeric_fp8 = true;
        PROOF_UPLOADS.with(|calls| calls.borrow_mut().clear());
        let resident = bindings.linear(&ops, &linear).unwrap();
        assert!(Rc::ptr_eq(
            &resident,
            &bindings.linear(&ops, &linear).unwrap()
        ));
        PROOF_UPLOADS.with(|calls| assert_eq!(&*calls.borrow(), &[addresses]));
        assert_eq!(counts(), (before.0 + 1, before.1 + 8));
        assert_eq!(ops.counters().artifact_uploads, 1);
        let super::super::binding::ResidentLinearStorage::Numeric(binding) = &resident.storage
        else {
            panic!("numeric device storage required")
        };
        assert_eq!(
            binding.artifact.layout(),
            artifact.validated_payload().layout()
        );
        // Raw callers remain untrusted: invalid bytes fail before allocation.
        ops.reset_counters();
        for bad in [0x7f, 0xff] {
            assert!(
                ops.upload_numeric_fp8_linear(
                    binding.artifact.layout(),
                    &[0x38, 0x38, 0x38, bad],
                    artifact.scale_bytes()
                )
                .is_err()
            );
        }
        for scale in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::MAX] {
            assert!(
                ops.upload_numeric_fp8_linear(
                    binding.artifact.layout(),
                    &[0x7e; 4],
                    &scale.to_le_bytes()
                )
                .is_err()
            );
        }
        assert!(
            ops.upload_numeric_fp8_linear(
                binding.artifact.layout(),
                &[0x38; 3],
                artifact.scale_bytes()
            )
            .is_err()
        );
        assert_eq!(ops.counters().device_allocation_attempts, 0);
        assert_eq!(ops.counters().artifact_uploads, 0);
        let replacement = fixture.directory.join("replacement");
        std::fs::copy(&fixture.source.scale().path, &replacement).unwrap();
        std::fs::rename(replacement, &fixture.source.scale().path).unwrap();
        assert!(
            bindings.linear(&ops, &linear).is_err(),
            "cached proof must not hide stale provenance"
        );
        assert!(upload(&ops, artifact).is_err());
        PROOF_UPLOADS.with(|calls| assert_eq!(calls.borrow().len(), 1));
        ops.sync_stream().unwrap();
        ops.sync_upload_stream().unwrap();
    }

    #[test]
    #[ignore = "requires CUDA; real numeric submission followed by lost consumer evidence"]
    fn numeric_unknown_retains_workspace_payload_operands_and_pending_credit() {
        unknown_custody(NumericFp8Precision::Bf16RneF32Accumulate);
    }

    #[test]
    #[ignore = "requires CUDA; F32 numeric submission followed by lost consumer evidence"]
    fn numeric_f32_unknown_retains_workspace_payload_operands_and_pending_credit() {
        unknown_custody(NumericFp8Precision::F32Tf32x3);
    }

    fn unknown_custody(precision: NumericFp8Precision) {
        let path = std::env::temp_dir().join(format!("numeric-unknown-{}.bin", std::process::id()));
        let mut payload = vec![0x38; 96];
        payload.extend(0x3c00u16.to_le_bytes());
        std::fs::write(&path, &payload).unwrap();
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        let mut slices = Vec::new();
        for (i, role) in [
            TensorRole::RoutedExpertGate,
            TensorRole::RoutedExpertUp,
            TensorRole::RoutedExpertDown,
        ]
        .into_iter()
        .enumerate()
        {
            let name = format!("projection{i}");
            let module = ModulePath::new(&name).unwrap();
            let shape = if i == 2 { vec![8, 12] } else { vec![12, 8] };
            schema
                .register_with_role(
                    ParameterSpec::new(
                        ParameterId::new(i as u64 + 1),
                        module.clone(),
                        ParameterDType::F8E4M3,
                        shape.clone(),
                        ParameterResidency::expert(0, 0),
                    )
                    .unwrap()
                    .with_required_scale(ParameterDType::Bf16, [1, 1])
                    .unwrap(),
                    role.clone(),
                )
                .unwrap();
            mapper
                .insert(&name, NameMapping::weight(module.clone()))
                .unwrap();
            mapper
                .insert(format!("{name}_scale"), NameMapping::scale(module))
                .unwrap();
            slices.push(CheckpointTensorSlice {
                name: name.clone(),
                path: path.clone(),
                role: role.clone(),
                offset: 0,
                bytes: 96,
                shape,
                dtype: CheckpointDType::F8E4M3,
            });
            slices.push(CheckpointTensorSlice {
                name: format!("{name}_scale"),
                path: path.clone(),
                role,
                offset: 96,
                bytes: 2,
                shape: vec![1, 1],
                dtype: CheckpointDType::Bf16,
            });
        }
        let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
            .bind_slices(slices)
            .unwrap();
        let materializer = StateDictMaterializer::new(1024).unwrap();
        let linear = |i| {
            let p = bound.get_by_id(ParameterId::new(i)).unwrap();
            materializer.prepared_linear(p, p.role().clone()).unwrap()
        };
        let expert = PreparedSwiGlu::new(linear(1), linear(2), linear(3), None).unwrap();
        let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
        let baseline = ops.allocator_metrics().live_requested_bytes;
        let mut owner = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
            Rc::clone(&ops),
            bound.parameters(),
            ExpertCachePolicy::Bounded(ExpertCacheLimits {
                max_experts: 1,
                max_bytes: 8192,
            }),
            4096,
            precision,
        )
        .unwrap();
        assert_eq!(owner.numeric_fp8_precision(), Some(precision));
        assert_eq!(owner.backend_name(), precision.standard_backend_name());
        let input = owner
            .bind_rows(Rows::Host(
                HostRows::new(
                    RowsShape::new(1, 8).unwrap(),
                    RowsDType::F32,
                    None,
                    vec![0.1; 8],
                )
                .unwrap(),
            ))
            .unwrap();
        let result = owner.synchronous(|this| {
            this.expert_operation(&expert, 1, |this| {
                let result = this.linear_rows(expert.gate(), &input, None)?;
                let mut output = this.ops.zero_f32_buffer(12)?;
                this.hold_expert_buffer(&output)?;
                this.ops.saxpy_into(1.0, this.f32(&result)?, &mut output)?;
                // Lose the final consumer proof, after a real numeric GEMM and
                // its subsequent output consumer, while the expert lease is pinned.
                this.lose_next_consumer_proof = true;
                Ok(output)
            })
        });
        assert!(result.is_err());
        assert!(owner.needs_quarantine());
        let frozen = owner.expert_cache_stats().unwrap();
        assert_eq!(frozen.pending_upload_bytes, 294);
        assert_eq!(frozen.resident_experts, 0);
        assert_eq!(frozen.unknown_quarantine_bytes, frozen.charged_bytes());
        assert!(frozen.scratch_bytes > 4096);
        let workspace = owner.numeric_workspace_stats().unwrap();
        assert_eq!(workspace.submissions, 1);
        if matches!(
            precision,
            NumericFp8Precision::F32Tf32x3 | NumericFp8Precision::W8A16Bf16F32Accumulate
        ) {
            // Fused M1/Fused W8A16 read compressed weights directly and need
            // no globally allocated numeric workspace.
            assert_eq!(workspace.allocated_bytes, 0);
        } else {
            assert!(workspace.allocated_bytes > 0);
        }
        assert!(!owner.expert_hold.is_empty());
        assert_eq!(owner.resident_parameter_bytes(), 98);
        ops.sync_stream().unwrap();
        ops.sync_upload_stream().unwrap();
        assert!(owner.prepare_expert(&expert).is_err());
        assert!(owner.quiesce().is_err());
        assert!(owner.finish_cache_operation().is_err());
        assert_eq!(owner.numeric_workspace_stats().unwrap(), workspace);
        assert_eq!(owner.expert_cache_stats().unwrap(), frozen);
        drop(input);
        let retained = ops.allocator_metrics().live_requested_bytes;
        assert!(retained >= baseline + workspace.allocated_bytes + 98 + 8 * 4 + 12 * 4 * 2);
        drop(owner);
        ops.trim_device_allocator().unwrap();
        assert!(ops.allocator_metrics().live_requested_bytes >= retained);
        assert!(Rc::strong_count(&ops) >= 2);
        std::fs::remove_file(path).unwrap();
    }
}
