//! Owner-local residency admission and exact consumer-completion custody.

use ferrule_backend::cuda::operators::linear::CudaF32Buffer;
use ferrule_common::Result;

use super::{CudaStandardDecoderOperators, ExpertCacheKey, cuda_error};
use crate::nn::ParameterResidency;
use crate::support::TensorRole;
use crate::transformer::{PreparedSwiGlu, SwiGluScratchPlan};

impl CudaStandardDecoderOperators {
    /// No decoding or upload during sizing. Converted BF16 has F32 device cost.
    pub(super) fn expert_plan(
        &self,
        expert: &PreparedSwiGlu,
        rows: usize,
    ) -> Result<(ExpertCacheKey, usize, usize)> {
        let ParameterResidency::Expert { layer, expert: id } =
            expert.gate().parameter().residency()
        else {
            return Err(cuda_error(
                "bounded routed expert requires expert residency",
            ));
        };
        let projections = [expert.gate(), expert.up(), expert.down()];
        let roles = [
            TensorRole::RoutedExpertGate,
            TensorRole::RoutedExpertUp,
            TensorRole::RoutedExpertDown,
        ];
        let mut bytes = 0usize;
        for (linear, role) in projections.iter().zip(&roles) {
            self.bindings.validate(linear.parameter())?;
            if linear.role() != role
                || linear.parameter().role() != role
                || linear.parameter().residency() != expert.gate().parameter().residency()
                || linear.parameter().binding().id() != linear.parameter().canonical_id()
            {
                return Err(cuda_error(
                    "expert canonical identity/residency/role mismatch",
                ));
            }
            if linear.tensor_shard().is_some()
                || self.bindings.tensor.is_some()
                || linear.bias().is_some()
                || (linear.numeric_fp8().is_none()
                    && linear.weight()?.execution.activation_quantization.is_some())
            {
                return Err(cuda_error(
                    "bounded experts require bias-free TP1 projections in the selected profile",
                ));
            }
            let projection_bytes = if let Some(artifact) = linear.numeric_fp8() {
                super::numeric_binding::validate_linear(linear)?;
                usize::try_from(artifact.storage_bytes())
                    .map_err(|_| cuda_error("expert bytes overflow"))?
            } else {
                linear
                    .in_features()
                    .checked_mul(linear.out_features())
                    .and_then(|n| n.checked_mul(4))
                    .ok_or_else(|| cuda_error("expert bytes overflow"))?
            };
            bytes = bytes
                .checked_add(projection_bytes)
                .ok_or_else(|| cuda_error("expert bytes overflow"))?;
        }
        let scratch = SwiGluScratchPlan::for_prepared(expert, rows)?.total_bytes()?;
        Ok((
            ExpertCacheKey {
                image_generation: self.image_generation(),
                numeric_precision: self.numeric_fp8_precision(),
                layer: *layer,
                expert: *id,
                parameters: projections.map(|p| (p.parameter().canonical_id(), p.role().clone())),
            },
            bytes,
            scratch,
        ))
    }

    pub(super) fn metadata_plan(
        &self,
        metadata: &crate::transformer::ExpertMetadata,
        layer: usize,
        expert: usize,
        rows: usize,
        width: usize,
    ) -> Result<(ExpertCacheKey, usize, usize)> {
        metadata.validate_sources()?;
        if metadata.input_width() != width || metadata.output_width() != width {
            return Err(cuda_error("routed metadata input/output width mismatch"));
        }
        for parameter in metadata.parameters() {
            self.bindings.validate_metadata(parameter)?;
            if parameter.residency() != &(ParameterResidency::Expert { layer, expert }) {
                return Err(cuda_error(
                    "provider returned another routed expert metadata",
                ));
            }
        }
        let scratch = SwiGluScratchPlan::for_metadata(metadata, rows)?.total_bytes()?;
        Ok((
            ExpertCacheKey {
                image_generation: self.image_generation(),
                numeric_precision: self.numeric_fp8_precision(),
                layer,
                expert,
                parameters: metadata
                    .parameters()
                    .each_ref()
                    .map(|p| (p.canonical_id(), p.role().clone())),
            },
            metadata.expected_device_bytes(),
            scratch,
        ))
    }

    pub(super) fn preflight_bucket_numeric(
        &self,
        parameters: &[crate::transformer::BoundParameter; 3],
        rows: usize,
    ) -> Result<()> {
        use ferrule_backend::cuda::operators::linear::{
            NumericFp8Layout, NumericFp8LinearPlan, NumericFp8ScaleType,
        };
        for parameter in parameters {
            let shape = parameter.spec().shape();
            if rows
                .checked_mul(shape[0])
                .is_none_or(|n| n > i32::MAX as usize)
            {
                return Err(cuda_error("expert bucket exceeds i32 ABI"));
            }
            if let (Some(state), Some(encoding)) = (&self.numeric, parameter.numeric_fp8_encoding())
            {
                let scale_type = match encoding {
                    crate::checkpoint::NumericFp8Encoding::E4M3FnBlock128Bf16 => {
                        NumericFp8ScaleType::Bf16
                    }
                    crate::checkpoint::NumericFp8Encoding::E4M3FnBlock128F32 => {
                        NumericFp8ScaleType::F32
                    }
                };
                NumericFp8LinearPlan::new(
                    NumericFp8Layout {
                        n: shape[0],
                        k: shape[1],
                        row_origin: 0,
                        column_origin: 0,
                        scale_type,
                    },
                    rows,
                    state.stats.reserved_bytes,
                    state.precision.backend(),
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn preflight_expert_admission(
        &self,
        bytes: usize,
        scratch: usize,
        route: usize,
    ) -> Result<()> {
        let scratch = SwiGluScratchPlan::reserved_bytes(
            scratch,
            route,
            self.numeric.as_ref().map_or(0, |n| n.stats.reserved_bytes),
        )?;
        if let Some(cache) = &self.bounded_experts {
            cache.preflight(bytes, scratch)?;
        }
        Ok(())
    }

    pub(super) fn cache_scratch(&mut self, bytes: usize) -> Result<()> {
        let total = SwiGluScratchPlan::reserved_bytes(
            bytes,
            0,
            self.numeric.as_ref().map_or(0, |n| n.stats.reserved_bytes),
        )?;
        if let Some(cache) = &mut self.bounded_experts {
            cache.scratch(total, |key| self.bindings.evict_expert(key))?;
        }
        if let Some(numeric) = &mut self.numeric {
            numeric.operation_bytes = bytes;
        }
        Ok(())
    }

    pub(super) fn reserve_numeric_temporaries(&mut self, bytes: usize) -> Result<()> {
        if self
            .numeric
            .as_ref()
            .is_some_and(|n| n.operation_bytes < bytes)
        {
            self.cache_scratch(bytes)?;
        }
        Ok(())
    }

    pub(super) fn cache_quarantine(&mut self) {
        if let Some(cache) = &mut self.bounded_experts {
            cache.quarantine();
        }
    }

    /// All views share allocations; they do not copy activations or extend a
    /// host weight cache. They survive error/unwind until a proven event.
    pub(super) fn hold_expert_buffer(&mut self, buffer: &CudaF32Buffer) -> Result<()> {
        if self.bindings.expert_admission.is_some()
            || self.numeric.is_some()
            || self.route_scratch_bytes != 0
        {
            self.expert_hold
                .push(buffer.as_device_buffer().slice(0, buffer.len())?);
        }
        Ok(())
    }
    pub(super) fn hold_route_buffer(&mut self, buffer: &CudaF32Buffer) -> Result<()> {
        if self.bounded_experts.is_some() || self.numeric.is_some() || self.route_scratch_bytes != 0
        {
            self.route_hold
                .push(buffer.as_device_buffer().slice(0, buffer.len())?);
        }
        Ok(())
    }

    pub(super) fn consumer_proof(&mut self) -> Result<()> {
        if self.bounded_experts.is_none() && self.numeric.is_none() && self.route_scratch_bytes == 0
        {
            return Ok(());
        }
        if self.needs_quarantine() {
            return Err(cuda_error("expert completion remains unknown"));
        }
        let compute = self
            .ops
            .record_compute_event()
            .and_then(|event| event.synchronize());
        let upload = self.ops.sync_upload_stream();
        #[cfg(test)]
        let compute = if std::mem::take(&mut self.lose_next_consumer_proof) {
            Err(cuda_error("injected lost consumer completion proof"))
        } else {
            compute
        };
        if let Err(error) = compute.and(upload) {
            self.poisoned = true;
            self.cache_quarantine();
            return Err(error);
        }
        Ok(())
    }

    pub(super) fn expert_operation<T>(
        &mut self,
        expert: &PreparedSwiGlu,
        rows: usize,
        run: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if self.numeric.is_some() {
            let bytes = SwiGluScratchPlan::for_prepared(expert, rows)?
                .total_with(self.route_scratch_bytes, 0)?;
            self.reserve_numeric_temporaries(bytes)?;
        }
        if self.bounded_experts.is_none() {
            return run(self);
        }
        if !matches!(
            expert.gate().parameter().residency(),
            ParameterResidency::Expert { .. }
        ) {
            // Shared and dense FFNs retain the original policy, not this cache.
            return run(self);
        }
        let (key, bytes, scratch) = self.expert_plan(expert, rows)?;
        self.admitted_expert_operation(key, bytes, scratch, run)
    }

    pub(super) fn admitted_expert_operation<T>(
        &mut self,
        key: ExpertCacheKey,
        bytes: usize,
        scratch: usize,
        run: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.preflight_expert_admission(bytes, scratch, self.route_scratch_bytes)?;
        let scratch = SwiGluScratchPlan::reserved_bytes(scratch, self.route_scratch_bytes, 0)?;
        self.cache_scratch(scratch)?;
        if self.bounded_experts.is_none() {
            return run(self);
        }
        let lease =
            self.bounded_experts
                .as_mut()
                .expect("bounded")
                .acquire(key.clone(), bytes, |key| self.bindings.evict_expert(key))?;
        self.bindings.expert_admission = Some(key);
        let result = run(self);
        // This event is on the actual matmul consumer stream, not an upload
        // completion event. Failed recording is unknown even if a later sync works.
        if let Err(error) = self.consumer_proof() {
            if let Ok(value) = result {
                std::mem::forget(value);
            }
            return Err(error);
        }
        self.expert_hold.clear();
        self.bindings.expert_admission = None;
        if result.is_err() {
            self.bindings.evict_expert(&lease.key);
        }
        self.bounded_experts
            .as_mut()
            .expect("bounded")
            .complete(&lease, result.is_ok())?;
        result
    }

    pub(super) fn finish_cache_operation(&mut self) -> Result<()> {
        if self.needs_quarantine() {
            return Err(cuda_error("cannot release unknown expert scratch"));
        }
        self.expert_hold.clear();
        self.route_hold.clear();
        self.route_ids.clear();
        self.route_scratch_bytes = 0;
        self.cache_scratch(0)
    }

    pub(super) fn retain_cache_on_unknown(&mut self) {
        self.cache_quarantine();
        std::mem::forget(std::mem::take(&mut self.expert_hold));
        std::mem::forget(std::mem::take(&mut self.route_hold));
        std::mem::forget(std::mem::take(&mut self.route_ids));
        std::mem::forget(self.bounded_experts.take());
        std::mem::forget(self.numeric.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ExecutionPrecisionPolicy;
    use crate::transformer::{ExpertCacheLimits, ExpertCachePolicy};
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use std::rc::Rc;

    #[test]
    #[ignore = "requires CUDA GPU; injects lost evidence, not a failing CUDA kernel"]
    fn bounded_cuda_unknown_keeps_scratch_credit_and_allocation_after_drop() {
        let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
        let mut owner = CudaStandardDecoderOperators::new_with_expert_cache(
            Rc::clone(&ops),
            ExecutionPrecisionPolicy::f32(),
            &[],
            ExpertCachePolicy::Bounded(ExpertCacheLimits {
                max_experts: 1,
                max_bytes: 16,
            }),
        )
        .unwrap();
        let baseline = ops.allocator_metrics().live_requested_bytes;
        owner.lose_next_consumer_proof = true;
        let result = owner.synchronous(|this| {
            this.cache_scratch(16)?;
            let scratch = this.ops.zero_f32_buffer(4)?;
            this.hold_route_buffer(&scratch)?;
            // The local handle drops before the fence; the owner's allocation
            // view must protect it even on an error or unwind path.
            drop(scratch);
            Ok(())
        });
        assert!(result.is_err());
        assert!(owner.needs_quarantine());
        let frozen = owner.expert_cache_stats().unwrap();
        assert_eq!(frozen.scratch_bytes, 16);
        assert_eq!(frozen.unknown_quarantine_bytes, 16);
        assert_eq!(owner.route_hold.len(), 1);
        ops.sync_stream().unwrap();
        ops.sync_upload_stream().unwrap();
        assert!(owner.quiesce().is_err());
        assert!(owner.finish_cache_operation().is_err());
        assert_eq!(owner.expert_cache_stats().unwrap(), frozen);
        drop(owner);
        ops.trim_device_allocator().unwrap();
        assert!(ops.allocator_metrics().live_requested_bytes >= baseline + 16);
        // Unknown owner deliberately retains a context authority on Drop.
        assert!(Rc::strong_count(&ops) >= 2);
    }
}
