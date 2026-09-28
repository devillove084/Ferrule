//! Explicit numeric FP8 storage profile and one fenced, reusable workspace.
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
    pub allocations: u64,
    pub reuses: u64,
    pub submissions: u64,
}

pub(super) struct NumericState {
    pub precision: NumericFp8Precision,
    pub workspace: Option<CudaNumericFp8Workspace>,
    pub stats: NumericWorkspaceStats,
    pub operation_bytes: usize,
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
    /// remain compressed in both modes. Scratch (including F32 decoded tiles)
    /// is bounded by the same reservation and reused only after consumer proof.
    pub fn new_numeric_fp8_with_precision(
        ops: Rc<CudaOperators>,
        parameters: &[BoundParameter],
        policy: ExpertCachePolicy,
        numeric_scratch_bytes: usize,
        precision: NumericFp8Precision,
    ) -> Result<Self> {
        if numeric_scratch_bytes == 0 {
            return Err(cuda_error("numeric scratch budget must be nonzero"));
        }
        let mut owner = Self::new_with_expert_profile(
            ops,
            crate::execution::ExecutionPrecisionPolicy::f32(),
            parameters,
            policy,
            true,
        )?;
        owner.numeric = Some(NumericState {
            precision,
            workspace: None,
            operation_bytes: 0,
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
        self.f32(input)?;
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
        let plan = NumericFp8LinearPlan::new(
            binding.artifact.layout(),
            input.shape().rows(),
            state.stats.reserved_bytes,
            state.precision.backend(),
        )?;
        // The old workspace is reusable only after an exact consumer proof.
        self.consumer_proof()?;
        let required = plan.workspace_requirements().bytes as usize;
        let need_alloc = self
            .numeric
            .as_ref()
            .and_then(|state| state.workspace.as_ref())
            .is_none_or(|workspace| workspace.allocated_bytes() < required);
        if need_alloc {
            let state = self.numeric.as_mut().expect("numeric profile");
            state.workspace = None;
            state.stats.allocated_bytes = 0;
            let workspace = self.ops.numeric_fp8_linear_workspace(plan)?;
            let state = self.numeric.as_mut().expect("numeric profile");
            state.workspace = Some(workspace);
            state.stats.allocated_bytes = required;
            state.stats.allocations += 1;
        } else {
            self.numeric.as_mut().expect("numeric profile").stats.reuses += 1;
        }
        let input = input
            .cuda()?
            .f32_buffer()
            .ok_or_else(|| cuda_error("numeric F32 input required"))?;
        let state = self.numeric.as_mut().expect("numeric profile");
        let workspace = state.workspace.as_mut().expect("numeric workspace");
        let result = self.ops.numeric_fp8_linear_into(
            &binding.artifact,
            input,
            output,
            workspace,
            plan,
            binding.artifact.layout().k,
            binding.artifact.layout().n,
        );
        state.stats.submissions += 1;
        if workspace.is_poisoned() {
            self.poisoned = true;
            self.cache_quarantine();
        }
        result?;
        self.consumer_proof()
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
        assert!(workspace.allocated_bytes > 0);
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
