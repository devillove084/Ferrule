//! Bounded submission of semantic operators; this is not a second forward.
//!
//! The layer caller must enter lexically around execute_standard_layer_inner and
//! flush before invoking external callbacks. Unclassified operators fence by
//! default. No reference to a numeric precision or provider is needed here.

use super::root_submission::{BarrierReason, SubmissionSegment};
use super::*;

#[derive(Default)]
pub(super) struct PreparedBindings {
    pub linears: std::collections::BTreeSet<(crate::nn::ParameterId, crate::support::TensorRole)>,
    norms: std::collections::BTreeSet<crate::nn::ParameterId>,
    ropes: Vec<PreparedRope>,
}

pub(super) fn operand_bytes(elements: usize, copies: usize) -> Result<usize> {
    elements
        .checked_mul(copies)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| cuda_error("root operand byte overflow"))
}

impl CudaStandardDecoderOperators {
    /// `pin_bytes` is a SUBSET of the caller's admitted activation workspace,
    /// not an additional allocation allowance. Numeric owners charge it against
    /// their existing cache limit as well. A scope cannot span layer/KV leave,
    /// recurrent state cloning or a public return. External callbacks require a
    /// producer barrier before dispatch.
    pub(super) fn with_root_submission<T>(
        &mut self,
        pin_bytes: usize,
        run: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if self.root_submission.is_some() || self.needs_quarantine() {
            return Err(cuda_error("nested or quarantined root submission"));
        }
        if self.ops.is_capture_safe() {
            return Err(cuda_error(
                "root submission requires preallocated capture support",
            ));
        }
        // The legacy collective and observer APIs can run arbitrary host code.
        // They stay synchronous until their callers supply explicit barriers.
        if self.bindings.tensor.is_some() || self.diagnostic.is_enabled() || self.bf16.is_some() {
            // BF16 operators own typed operands and fence each semantic boundary.
            return self.synchronous(run);
        }
        let segment = SubmissionSegment::new(pin_bytes)
            .ok_or_else(|| cuda_error("root submission requires an admitted nonzero pin budget"))?;
        if !self.expert_hold.is_empty()
            || !self.route_hold.is_empty()
            || !self.expert_artifacts.is_empty()
            || !self.route_ids.is_empty()
            || self.bindings.expert_admission.is_some()
        {
            return Err(cuda_error(
                "root submission overlaps another operation's custody",
            ));
        }
        // This also rejects capture started through the lower-level graph API,
        // which need not have enabled CudaOperators' capture-safe assertion.
        if let Err(error) = self.ops.sync_upload_stream() {
            self.poison_root_submission();
            return Err(error);
        }
        self.cache_scratch(pin_bytes)?;
        self.root_submission = Some(segment);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(self)));
        match result {
            Ok(Ok(value)) => {
                if let Err(error) = self.root_barrier(BarrierReason::SegmentExit) {
                    std::mem::forget(value);
                    return Err(error);
                }
                self.root_submission = None;
                self.finish_cache_operation()?;
                Ok(value)
            }
            Ok(Err(error)) => {
                // External dispatch is different from a failed local operation:
                // ExternalExperts retires every local root before the callback,
                // so its ordinary executor error remains drainable by the owner.
                // A failed producer proof quarantines first and cannot enter this
                // branch because `needs_quarantine()` is then true.
                let producer_complete = !self.needs_quarantine()
                    && self
                        .root_submission
                        .as_ref()
                        .is_some_and(SubmissionSegment::can_return_external_error);
                if producer_complete {
                    self.root_submission = None;
                    if let Err(cleanup) = self.finish_cache_operation() {
                        self.poison_root_submission();
                        return Err(crate::transformer::parallel_transfer::CudaShardError::wrap(
                            error,
                            Some(cleanup),
                            false,
                            true,
                        ));
                    }
                    return Err(error);
                }
                // A failed local submission is not evidence that no kernel
                // launched. Keep the original typed error and all roots/credits
                // sticky.
                self.poison_root_submission();
                Err(error)
            }
            Err(panic) => {
                self.poison_root_submission();
                std::panic::resume_unwind(panic)
            }
        }
    }

    pub(super) fn poison_root_submission(&mut self) {
        if let Some(segment) = &mut self.root_submission {
            segment.quarantine();
        }
        self.poisoned = true;
        self.cache_quarantine();
    }

    /// Explicit host/ownership/storage boundary. Never uses an allocation event
    /// as consumer evidence and never releases roots if ANY proof is unknown.
    pub(in crate::transformer) fn root_barrier(&mut self, reason: BarrierReason) -> Result<()> {
        let Some(segment) = &self.root_submission else {
            return Ok(());
        };
        debug_assert!(segment.bytes() <= segment.limit());
        if segment.is_unknown() || self.needs_quarantine() {
            return Err(cuda_error("root submission completion remains unknown"));
        }
        if !segment.has_pending() {
            // For example, a synchronous shared gate has already proved local
            // completion before external dispatch. Conversely, OpaqueOperator
            // must revoke a preceding external handoff before running locally.
            self.root_submission
                .as_mut()
                .unwrap()
                .note_idle_barrier(reason);
            return Ok(());
        }
        // Growth inside a projection must fence, but cannot retire its input or
        // output: that operator may submit another kernel after this boundary.
        let submitting = segment.can_defer_ordered_proof();
        let event = self.consumer_proof();
        let drain = self.boundary.drain();
        let compute = self.ops.sync_stream();
        let upload = self.ops.sync_upload_stream();
        if let Err(error) = event.and(drain).and(compute).and(upload) {
            self.poison_root_submission();
            return Err(error);
        }
        if !submitting {
            if !self
                .root_submission
                .as_mut()
                .unwrap()
                .retire_after_proof(reason)
            {
                self.poison_root_submission();
                return Err(cuda_error("root retirement without completion authority"));
            }
            self.expert_hold.clear();
            self.expert_artifacts.clear();
            self.route_hold.clear();
            self.route_ids.clear();
        }
        Ok(())
    }

    pub(super) fn root_pin(&mut self, buffer: &CudaF32Buffer) -> Result<()> {
        let Some(segment) = &mut self.root_submission else {
            return Ok(());
        };
        if !segment.can_defer_ordered_proof() {
            return Ok(());
        }
        let root = buffer.as_device_buffer().slice(0, buffer.len())?;
        let key = root.cu_deviceptr();
        let bytes = root.num_bytes();
        segment
            .pin(key, bytes, root)
            .map_err(|_| cuda_error("root pin exceeds preflighted budget"))
    }

    pub(super) fn root_pin_rows(&mut self, rows: &Rows) -> Result<()> {
        if !self.root_is_submitting() {
            return Ok(());
        }
        let buffer = self.f32(rows)?;
        let root = buffer.as_device_buffer().slice(0, buffer.len())?;
        if let Some(segment) = &mut self.root_submission
            && segment.can_defer_ordered_proof()
        {
            let key = root.cu_deviceptr();
            let bytes = root.num_bytes();
            segment
                .pin(key, bytes, root)
                .map_err(|_| cuda_error("root operand exceeds pin budget"))?;
        }
        Ok(())
    }

    /// One finite semantic operation. Forecast includes every input and newly
    /// allocated output (duplicates may be overcharged, never undercharged).
    pub(super) fn root_operation<T>(
        &mut self,
        bytes: usize,
        roots: usize,
        run: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if self.diagnostic.is_enabled() {
            self.root_barrier(BarrierReason::Diagnostic)?;
            return self.synchronous(run);
        }
        let Some(segment) = &self.root_submission else {
            return self.synchronous(run);
        };
        if self.needs_quarantine() {
            return Err(cuda_error("root submission requires quarantine"));
        }
        if !segment.can_fit(bytes, roots) {
            self.root_barrier(BarrierReason::PinBudget)?;
        }
        if !self.root_submission.as_ref().unwrap().can_fit(bytes, roots) {
            // Larger operations retain their ordinary one-operation lifetime.
            return self.synchronous(run);
        }
        if !self.root_submission.as_mut().unwrap().begin_operation() {
            return Err(cuda_error("overlapping root operator"));
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(self)));
        match result {
            Ok(Ok(value)) if !self.needs_quarantine() => {
                if !self.root_submission.as_mut().unwrap().finish_operation() {
                    std::mem::forget(value);
                    self.poison_root_submission();
                    return Err(cuda_error("root operator lost submission custody"));
                }
                Ok(value)
            }
            Ok(result) => {
                self.poison_root_submission();
                match result {
                    Ok(value) => {
                        std::mem::forget(value);
                        Err(cuda_error("root completion unknown"))
                    }
                    Err(error) => Err(error),
                }
            }
            Err(panic) => {
                self.poison_root_submission();
                std::panic::resume_unwind(panic)
            }
        }
    }

    pub(super) fn ordered_consumer_proof(&mut self) -> Result<()> {
        if self.needs_quarantine() {
            return Err(cuda_error("ordered consumer completion remains unknown"));
        }
        if self
            .root_submission
            .as_ref()
            .is_some_and(|s| s.can_defer_ordered_proof())
        {
            return Ok(());
        }
        self.consumer_proof()
    }
}

impl CudaStandardDecoderOperators {
    pub(super) fn root_is_submitting(&self) -> bool {
        self.root_submission
            .as_ref()
            .is_some_and(|segment| segment.can_defer_ordered_proof())
    }

    pub(super) fn root_norm_binding(&mut self, norm: &PreparedNorm) -> Result<Rc<CudaF32Buffer>> {
        let id = norm.parameter().canonical_id();
        if !self.root_bindings.norms.contains(&id) {
            self.root_barrier(BarrierReason::CacheEvictionOrRebind)?;
        }
        let weight = self.bindings.norm(&self.ops, norm)?;
        self.root_bindings.norms.insert(id);
        Ok(weight)
    }

    pub(super) fn root_rope_binding(
        &mut self,
        table: &PreparedRope,
    ) -> Result<Rc<binding::ResidentRope>> {
        let known = self.root_bindings.ropes.contains(table);
        if !known {
            self.root_barrier(BarrierReason::CacheEvictionOrRebind)?;
        }
        let resident = self.bindings.rope(&self.ops, table)?;
        if !known {
            self.root_bindings.ropes.push(table.clone());
        }
        Ok(resident)
    }

    fn root_linear_resident(&self, linear: &PreparedLinear) -> Result<bool> {
        self.bindings.validate(linear.parameter())?;
        let key = (linear.parameter().canonical_id(), linear.role().clone());
        // Non-expert bindings are immutable for the image lifetime. This set is
        // populated only after Bindings::linear succeeds; it is not a freshness
        // witness and never suppresses source/bias validation in linear_rows.
        if self.root_bindings.linears.contains(&key) {
            return Ok(true);
        }
        Ok(self
            .bindings
            .metadata_linear(linear.parameter().binding())?
            .is_some())
    }

    pub(super) fn root_swiglu_resident(&self, expert: &PreparedSwiGlu) -> Result<bool> {
        for linear in [expert.gate(), expert.up(), expert.down()] {
            if linear.bias().is_some()
                || linear.tensor_shard().is_some()
                || linear.parameter().binding().is_alias()
                || !matches!(
                    linear.parameter().residency(),
                    crate::nn::ParameterResidency::Layer { .. }
                )
                || !self.root_linear_resident(linear)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(super) fn root_linear(
        &mut self,
        linear: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        let eligible = self.bindings.tensor.is_none()
            && !self.diagnostic.is_enabled()
            && linear.bias().is_none()
            && linear.tensor_shard().is_none()
            && !linear.parameter().binding().is_alias()
            && !matches!(
                linear.parameter().residency(),
                crate::nn::ParameterResidency::Expert { .. }
            );
        if !eligible {
            self.root_barrier(if self.bindings.tensor.is_some() {
                BarrierReason::TensorCollective
            } else {
                BarrierReason::OpaqueOperator
            })?;
            return self.synchronous(|this| {
                this.linear_rows(linear, input, arena)
                    .map(OperatorProgress::Ready)
            });
        }
        if !self.root_linear_resident(linear)? {
            self.root_barrier(BarrierReason::CacheEvictionOrRebind)?;
            return self.synchronous(|this| {
                this.linear_rows(linear, input, arena)
                    .map(OperatorProgress::Ready)
            });
        }
        let elements = linear
            .in_features()
            .checked_add(linear.out_features())
            .and_then(|n| n.checked_mul(input.shape().rows()))
            .ok_or_else(|| cuda_error("root projection size overflow"))?;
        let bytes = operand_bytes(elements, 1)?;
        if self.root_submission.is_none() {
            // Compatibility KeepAll has no redundant consumer events to merge
            // within a single projection. It benefits only at the layer seam.
            if self.bounded_experts.is_none() && self.numeric.is_none() {
                return self.synchronous(|this| {
                    this.linear_rows(linear, input, arena)
                        .map(OperatorProgress::Ready)
                });
            }
            return self.with_root_submission(bytes, |this| this.root_linear(linear, input, arena));
        }
        self.root_operation(bytes, 2, |this| {
            this.linear_rows(linear, input, arena)
                .map(OperatorProgress::Ready)
        })
    }
}

impl CudaStandardDecoderOperators {
    /// Concrete integration seam for the model-parallel owner. Uses precisely
    /// the existing layer implementation, never an alternate attention/FFN loop.
    /// Shared composition fences before external executor/liveness callbacks.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::transformer) fn execute_root_layer(
        &mut self,
        materializer: &crate::transformer::StateDictMaterializer,
        layer_index: usize,
        attention: super::super::hybrid::AttentionBlockRef<'_>,
        feed_forward: &super::super::PreparedFeedForwardBlock,
        kv_layer: usize,
        hidden: &mut super::super::CpuTransformerHidden,
        kv: &mut dyn KvView,
        states: &mut [&mut dyn crate::decoder::StandardSequenceState],
        routed_execution: Option<&mut super::super::RoutedLayerExecution<'_>>,
    ) -> Result<()> {
        use super::super::PreparedFeedForwardKind;
        use super::super::hybrid::AttentionBlockRef;
        if self.root_submission.is_some() {
            return Err(cuda_error(
                "root layer cannot nest inside a submission scope",
            ));
        }
        let rows = hidden.rows()?.shape().rows();
        // Cap delayed retirement at the largest ALREADY admitted operation,
        // not a sum of layer intermediates. Norms fit the existing 32*width
        // inventory envelope; linear/FFN plans match numeric admission exactly.
        let mut pin_bytes = operand_bytes(hidden.rows()?.shape().elements(), 2)?;
        let projections: &[&PreparedLinear] = match &attention {
            AttentionBlockRef::Gqa(a) => &[&a.query, &a.key, &a.value, &a.output],
            AttentionBlockRef::GatedDeltaNet(a) => &[&a.qkv, &a.z, &a.beta, &a.a, &a.output],
        };
        for linear in projections {
            let elements = linear
                .in_features()
                .checked_add(linear.out_features())
                .and_then(|n| n.checked_mul(rows))
                .ok_or_else(|| cuda_error("root layer pin budget overflow"))?;
            pin_bytes = pin_bytes.max(operand_bytes(elements, 1)?);
        }
        let shared = match &feed_forward.kind {
            PreparedFeedForwardKind::Dense(expert) => Some(expert),
            PreparedFeedForwardKind::Routed { shared, .. } => shared.as_ref(),
        };
        if let Some(expert) = shared {
            pin_bytes = pin_bytes.max(expert.f32_scratch_bytes(rows)?);
        }
        let run = |this: &mut Self| {
            super::super::execute_standard_layer_inner(
                this,
                materializer,
                layer_index,
                attention,
                feed_forward,
                kv_layer,
                hidden,
                kv,
                states,
                routed_execution,
            )
        };
        self.with_root_submission(pin_bytes, run)
    }
}
