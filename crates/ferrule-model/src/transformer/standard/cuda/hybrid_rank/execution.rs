//! Owner orchestration only: the shared segment owns the one layer loop.
use super::*;

impl CudaHybridRankSegment {
    /// Host-boundary convenience API. Hidden input is uploaded once; the
    /// layer loop and all model arithmetic remain on CUDA. State/KV publication
    /// is still owned by the caller's journal transaction.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [CudaHybridRankState],
        kv: &mut CudaKvView,
        sequence_ids: &[u64],
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<SegmentOutput> {
        let output =
            self.execute_cuda_bound(batch, input.into(), states, kv, sequence_ids, check_active)?;
        let output = self.download_output(output)?;
        if let Err(error) = check_active(kv.transaction()) {
            self.failed = true;
            self.channel.borrow_mut().abort();
            return Err(error);
        }
        Ok(output)
    }

    /// Execute one entered journal transaction. Session IDs are in packed
    /// sequence order. This neither commits state nor publishes physical KV.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cuda_bound(
        &mut self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        states: &mut [CudaHybridRankState],
        kv: &mut CudaKvView,
        sequence_ids: &[u64],
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<DeviceSegmentOutput> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.device.ensure_ready()?;
            if self.closed
                || self.shutdown_started
                || self.failed
                || self.external.forward.is_some()
                || self.watermark.is_some_and(|last| last >= kv.transaction())
                || batch.len() > self.memory.max_rows
                || batch.sequences().len() > self.memory.max_sequences
                || sequence_ids.len() != batch.sequences().len()
                || batch.page_size() != self.memory.page_size
                || batch
                    .positions()
                    .iter()
                    .any(|&p| p >= self.memory.max_positions)
            {
                return Err(cuda_error(
                    "CUDA rank closed/stale/failed or batch exceeds admission",
                ));
            }
            check_active(kv.transaction())?;
            for state in states.iter() {
                state.validate_rank(&self.state_credits)?;
            }
            self.segment
                .validate_execution(batch, states)
                .map_err(segment_error)?;
            kv.validate_batch(batch)?;
            // Check every used state and compact plane before any device write.
            for sequence in batch.sequences() {
                for state in states[sequence.state_index()]
                    .kv_state()
                    .layers()
                    .iter()
                    .flatten()
                {
                    state.validate_owner(self.device.operators())?;
                }
            }
            for layer in self.schema().layers() {
                if let HybridLayerSchema::FullAttention {
                    kv_layer,
                    kv_heads,
                    head_dim,
                } = layer
                {
                    kv.with_f32_gqa_planes(
                        self.device.operators(),
                        *kv_layer,
                        batch,
                        batch.protected_pages(),
                        |planes| {
                            let stride = planes
                                .layout
                                .checked_elements_per_page()
                                .ok_or_else(|| cuda_error("rank KV stride overflow"))?;
                            let kv_layers = self
                                .schema()
                                .layers()
                                .iter()
                                .filter(|l| matches!(l, HybridLayerSchema::FullAttention { .. }))
                                .count();
                            if planes.layout.elements_per_token != kv_heads * head_dim
                                || planes.layout.layer_count != kv_layers
                                || planes.layout.page_tokens != self.memory.page_size
                                || stride.checked_mul(self.memory.physical_pages)
                                    != Some(planes.key.len())
                                || planes.key.len() != planes.value.len()
                            {
                                return Err(cuda_error("rank compact KV plane geometry mismatch"));
                            }
                            Ok(())
                        },
                    )?;
                }
            }
            // V2 agrees on logical identities, never owner-local generations.
            // Hidden staging is transport only; retain the original device input.
            let agreement_input = match &input {
                DeviceSegmentInput::Tokens => SegmentInput::Tokens,
                DeviceSegmentInput::Hidden { next_layer, rows } => {
                    let values = match rows {
                        Rows::Host(rows) => rows.values().to_vec(),
                        Rows::Cuda(rows) => {
                            rows.validate_owner(self.device.operators())?;
                            let buffer = rows
                                .f32_buffer()
                                .ok_or_else(|| cuda_error("rank input requires F32"))?;
                            self.operators
                                .root_barrier(BarrierReason::TensorCollective)?;
                            self.operators
                                .boundary
                                .download(self.device.operators(), buffer)?
                        }
                    };
                    SegmentInput::Hidden {
                        next_layer: *next_layer,
                        rows: HostRows::new(rows.shape(), rows.dtype(), rows.arena(), values)?,
                    }
                }
            };
            let identity = self.channel.borrow().schedule.encode_agreement(
                batch,
                sequence_ids,
                &agreement_input,
            )?;
            self.watermark = Some(kv.transaction());
            self.channel.borrow_mut().begin(
                kv.transaction(),
                batch.len(),
                batch.sequences().len(),
            )?;
            let gathered = self.channel.borrow_mut().exchange(
                HybridCollectiveSemantic::Agreement,
                None,
                identity.clone(),
            )?;
            if gathered
                .chunks_exact(identity.len())
                .any(|part| part != identity)
            {
                return Err(cuda_error("TP lanes disagree on V2 input identity"));
            }
            self.external.begin(kv.transaction(), batch, kv)?;
            self.external.kv = Some(kv.clone());
            let source = self.parallel.expert_attachment().map(|ep| ep.source_rank);
            let transaction = kv.transaction();
            let mut binding = CudaStandardKvBinding::new(kv, batch)?;
            let (call, external_check) = match self.external.attachment.as_mut() {
                Some(experts) => (
                    Some(RoutedCall {
                        executor: experts
                            .executor
                            .as_mut()
                            .expect("attached executor")
                            .as_mut(),
                        device: &self.device,
                        holds: &mut self.external.holds,
                    }),
                    experts.check_active.as_mut(),
                ),
                None => (None, None),
            };
            let mut external_check = external_check;
            let mut active = |id| {
                self.device.ensure_ready()?;
                if let Some(forward) = &self.external.forward {
                    forward.check(id, batch)?;
                }
                check_active(id)?;
                if let Some(check) = &mut external_check {
                    check(id)?;
                }
                Ok(())
            };
            let mut routed = RankExperts {
                call,
                device: &self.device,
                boundary: &mut self.expert_boundary,
                channel: &self.channel,
                parallel: &self.parallel,
                owner: self.layout.owner(),
            };
            let output = self
                .segment
                .execute_bound_inner(
                    batch,
                    input,
                    states,
                    &mut binding,
                    crate::transformer::standard::LayerOperators::Cuda(&mut self.operators),
                    &mut active,
                    source.map(|source| {
                        (
                            (transaction, source, sequence_ids),
                            &mut routed as &mut dyn RoutedSwiGluExecutor,
                        )
                    }),
                )
                .map_err(segment_error)?;
            let completion = (|| {
                self.external.require_drain(&self.device, false)?;
                self.operators.quiesce()?;
                self.expert_boundary.drain()?;
                self.device.fence()?;
                self.external.check(&self.device, kv.transaction(), batch)?;
                check_active(kv.transaction())?;
                self.segment
                    .validate_frontier(batch, states)
                    .map_err(segment_error)?;
                let completed = self.channel.borrow_mut().exchange(
                    HybridCollectiveSemantic::Completion,
                    None,
                    vec![1.],
                )?;
                if completed != [self.layout.members().len() as f32] {
                    return Err(cuda_error("rank completion agreement mismatch"));
                }
                self.channel.borrow_mut().finish()
            })();
            if let Err(error) = completion {
                if self.needs_quarantine() {
                    std::mem::forget(output);
                }
                return Err(error);
            }
            self.external.finish();
            Ok(output)
        }));
        if self.operators.needs_quarantine() || self.expert_boundary.needs_quarantine() {
            self.device.quarantine();
        }
        match result {
            Ok(Ok(output)) => Ok(output),
            Ok(Err(primary)) => {
                self.failed = true;
                self.channel.borrow_mut().abort();
                Err(Error::with_cleanup(
                    "CUDA rank execution",
                    primary,
                    self.external.fail(),
                ))
            }
            Err(panic) => {
                self.failed = true;
                self.channel.borrow_mut().abort();
                self.device.quarantine();
                self.operators.poisoned = true;
                std::panic::resume_unwind(panic)
            }
        }
    }

    pub fn download_output(&mut self, output: DeviceSegmentOutput) -> Result<SegmentOutput> {
        let result = output.into_host(&mut self.operators).map_err(segment_error);
        if self.operators.needs_quarantine() {
            self.device.quarantine();
        }
        if result.is_err() {
            self.failed = true;
            self.channel.borrow_mut().abort();
        }
        result
    }
}

struct RankExperts<'a> {
    call: Option<RoutedCall<'a>>,
    device: &'a HybridCudaDevice,
    boundary: &'a mut Boundary,
    channel: &'a RefCell<RankChannel>,
    parallel: &'a HybridParallelPlan,
    owner: ParallelRankId,
}
impl RoutedSwiGluExecutor for RankExperts<'_> {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        active(request.context.transaction)?;
        if Some(request.context.source_rank)
            != self.parallel.expert_attachment().map(|ep| ep.source_rank)
            || !self
                .parallel
                .schema()
                .layer_range()
                .contains(&request.context.layer)
            || self.call.is_some() != (self.owner == request.context.source_rank)
            || request.sequences.len() != request.input.shape().rows()
        {
            return Err(cuda_error("rank routed source/layer/rows mismatch"));
        }
        let layer = request.context.layer;
        let shape = request.input.shape();
        let arena = request.arena;
        let values = if let Some(call) = &mut self.call {
            // A waiting dispatch is NOT replayed. The outer owner retains the
            // shared external custody and requires cancellation/drain proof.
            let rows = crate::transformer::standard::ready(call.routed_swiglu(request, active)?)?;
            let rows = rows.cuda()?;
            rows.validate_owner(self.device.operators())?;
            let buffer = rows
                .f32_buffer()
                .ok_or_else(|| cuda_error("EP output requires F32"))?;
            self.boundary.download(self.device.operators(), buffer)?
        } else {
            // This is the additive transport identity, not CPU model math.
            vec![0.; shape.elements()]
        };
        let values = self.channel.borrow_mut().exchange(
            HybridCollectiveSemantic::RoutedExpertOutput,
            Some(layer),
            values,
        )?;
        let buffer = self.boundary.upload(self.device.operators(), &values)?;
        device_rows(shape, arena, buffer).map(OperatorProgress::Ready)
    }
}
