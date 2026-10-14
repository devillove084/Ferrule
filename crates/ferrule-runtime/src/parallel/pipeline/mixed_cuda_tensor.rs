//! Mixed owner adapter only. The model owns arithmetic and the worker owns KV/state commit.
use super::*;
use crate::parallel::tensor::decoder_collective::MixedRankCollective;
use ferrule_model::decoder::{CompletionUnknown, DecoderSequenceLifecycle, StandardSequenceState};
use ferrule_model::transformer::hybrid_parallel::HybridMixedRankProgramPlan;
use ferrule_model::transformer::{
    CudaStandardDecoderOperators, DeviceSegmentInput, ExpertCachePolicy,
    NumericFp8Precision, RowsDType, RowsShape,
};

use super::super::mixed_trace::{MixedTraceWriter, bounded_debug};
use std::cell::RefCell;

fn attention_partition(
    partition: &ferrule_model::transformer::hybrid_parallel::HybridLocalAttention,
) -> serde_json::Value {
    use ferrule_model::transformer::hybrid_parallel::HybridLocalAttention;
    use serde_json::json;
    match partition {
        HybridLocalAttention::Gqa {
            query_heads,
            kv_heads,
            kv_owners,
            head_dim,
        } => json!({
            "kind":"gqa", "query_heads":[query_heads.start,query_heads.end],
            "kv_heads":[kv_heads.start,kv_heads.end],
            "kv_owners":kv_owners.iter().map(|r|r.get()).collect::<Vec<_>>(),"head_dim":head_dim
        }),
        HybridLocalAttention::GatedDelta {
            key_heads,
            value_heads,
            qkv_channels,
            shape,
        } => json!({
            "kind":"gated_delta", "key_heads":[key_heads.start,key_heads.end],
            "value_heads":[value_heads.start,value_heads.end],
            "qkv_channels":qkv_channels.iter().map(|r|[r.start,r.end]).collect::<Vec<_>>(),
            "shape":{"key_heads":shape.key_heads,"value_heads":shape.value_heads,
                "key_dim":shape.key_dim,"value_dim":shape.value_dim,"kernel":shape.kernel}
        }),
    }
}

#[path = "mixed_cache_observation.rs"]
mod cache_observation;
use cache_observation::emit_cache_snapshot;

pub struct MixedCudaTensorProgram {
    trace: Option<Rc<RefCell<MixedTraceWriter>>>,
    plan: HybridMixedRankProgramPlan,
    operators: Option<CudaStandardDecoderOperators>,
    lifecycle: Option<CudaHybridRankLifecycle>,
    device: HybridCudaDevice,
    hidden: usize,
    failed: bool,
    closed: bool,
}
impl HybridCudaTensorLeaf for MixedCudaTensorProgram {
    const MIXED: bool = true;
}
impl MixedCudaTensorProgram {
    fn completion<T>(&self, result: Result<T>) -> Result<T> {
        checked_cuda_completion(
            self.device.needs_quarantine()
                || self
                    .operators
                    .as_ref()
                    .is_some_and(|o| o.needs_quarantine()),
            result,
        )
    }
}
impl PipelineStateProgram for MixedCudaTensorProgram {
    type SequenceState = CudaHybridRankState;
    type Lifecycle = CudaHybridRankLifecycle;
    type KvView = CudaKvView;
    fn plan(&self) -> &LayerSegmentPlan {
        self.plan.segment()
    }
    fn hybrid_schema(&self) -> Option<&HybridStateSchema> {
        Some(self.plan.layout().state_schema())
    }
    fn hybrid_precision(&self) -> ExecutionPrecisionPolicy {
        ExecutionPrecisionPolicy::bf16_compatibility()
    }
    fn new_lifecycle(&mut self) -> Result<Self::Lifecycle> {
        self.lifecycle
            .take()
            .ok_or_else(|| error("Mixed lifecycle already assigned to owner journal"))
    }
    fn validate_execution(
        &self,
        _: &KvCommitBinding,
        batch: &PackedDecoderBatch,
        sessions: &[u64],
    ) -> Result<()> {
        if self.closed || self.failed || self.operators.is_none() {
            return Err(error("Mixed owner is closed or failed; retry rejected"));
        }
        self.completion(self.plan.validate_input(
            batch,
            RowsShape::new(batch.len(), self.hidden)?,
            RowsDType::Bf16,
            sessions,
        ))
    }
    fn execute_state(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [Self::SequenceState],
        view: &mut CudaKvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        if self.closed
            || self.failed
            || context.experts.is_some()
            || context.transaction != view.transaction()
            || !matches!(input, SegmentInput::Tokens)
        {
            return Err(error("Mixed owner requires sealed local token segment"));
        }
        if let Some(trace) = &self.trace
            && trace.borrow().accepts_metadata(
                batch
                    .len()
                    .saturating_mul(2)
                    .saturating_add(batch.sequences().len().saturating_mul(16)),
                0,
            )
        {
            trace.borrow_mut().begin(context.transaction.get(), serde_json::json!({
                "owner":self.plan.layout().owner().get(), "transaction":context.transaction.get(),
                "sessions":context.sequence_ids,"tokens":batch.token_ids(),"positions":batch.positions(),
                "sequences":batch.sequences().iter().map(|s| serde_json::json!({
                    "phase":format!("{:?}",s.phase()),"context_len":s.context_len(),
                    "sequence_len":s.sequence_len(),"query":[s.query().start,s.query().end],
                    "page_generation":s.page_generation(),"execution_generation":s.execution_generation(),
                    "state_index":s.state_index()
                })).collect::<Vec<_>>()
            }));
        }
        let result = cuda_call(|| {
            (context.check_active)(context.transaction)?;
            let ops = self
                .operators
                .as_mut()
                .ok_or_else(|| error("Mixed owner operators absent"))?;
            let output = ops.execute_mixed_rank_segment(
                batch,
                DeviceSegmentInput::Tokens,
                states,
                view,
                context.sequence_ids,
                context.check_active,
            )?;
            // This is the final logits API boundary only. All rank exchange
            // contributions remain Vec<u16>, with no F32 compatibility wrapper.
            let output = output.into_host(ops).map_err(stage::segment_error)?;
            (context.check_active)(context.transaction)?;
            Ok(output)
        });
        if let Some(trace) = &self.trace {
            trace.borrow_mut().finish(result.is_ok());
        }
        if result.is_err() {
            self.failed = true;
        }
        self.completion(result)
    }
    fn validate_frontier(
        &self,
        batch: &PackedDecoderBatch,
        states: &[Self::SequenceState],
    ) -> Result<()> {
        let result = batch.sequences().iter().try_for_each(|s| {
            states
                .get(s.state_index())
                .ok_or_else(|| error("Mixed state frontier index"))?
                .validate_standard_state_at(self.plan.layout().state_schema(), s.sequence_len())
        });
        self.completion(result)
    }
    fn poll_quiescence(&mut self) -> Result<bool> {
        let result = cuda_call(|| match self.operators.as_mut() {
            Some(ops) => ops.quiesce(),
            None => Ok(()),
        });
        self.completion(result).map(|_| true)
    }
    fn trace_committed(
        &mut self,
        backend: &mut dyn std::any::Any,
        transaction: ferrule_common::execution::ExecutionTransactionId,
        source: &ferrule_common::host_kv::CommittedKvSource,
    ) -> Result<()> {
        use ferrule_common::execution::{KvElementType, KvLayoutSchema};
        use ferrule_model::transformer::hybrid_parallel::HybridLocalAttention;
        if self.closed || self.failed {
            return Err(error("committed trace owner closed/failed"));
        }
        let trace = self
            .trace
            .as_ref()
            .ok_or_else(|| error("native trace disabled"))?
            .clone();
        if trace.borrow().failed() {
            return Err(error("native capture already incomplete"));
        }
        let pool = backend
            .downcast_mut::<PagedKvBackend<TypedCudaPagedKvPool<CudaHybridRankState>>>()
            .ok_or_else(|| error("trace requires the owner's typed BF16 KV backend"))?;
        let config = trace.borrow().config().clone();
        let page_size = source.schema.page_size();
        let max_positions = source.schema.max_sequence_len();
        // Project global logical head geometry through the SAME sealed owner layout.
        // Page/slot/generation/incarnation/frontier remain from the retained parent lease.
        let planes = self.plan.layout().state_schema().kv_planes_with_dtype(
            page_size,
            max_positions,
            KvElementType::Bf16,
        )?;
        let mut local_source = source.clone();
        local_source.schema = planes.schema_identity();
        let words = config.max_tensor_bytes / 4;
        let export_host = ferrule_backend::cuda::operators::kv::committed_bf16_kv_trace_host_bytes(
            &local_source,
            words,
        )?;
        if export_host + config.max_tensor_bytes
            > config.host_reserve_with_pages(1, source.pages.len())? as usize
        {
            return Err(error("committed trace host allowance exceeded"));
        }
        trace.borrow_mut().committed_begin(
            transaction.get(),
            serde_json::json!({
                "transaction":transaction.get(),"source_owner":source.owner,
                "state_slot":source.state_slot.get(),"generation":source.generation,
                "incarnation":source.incarnation,"committed_tokens":source.committed_tokens,
                "page_size":page_size
            }),
        )?;
        let result = (|| {
            pool.bind_committed_logical_owner(source.owner)?;
            for layer in self.plan.layers() {
                let HybridLocalAttention::Gqa {
                    kv_heads,
                    head_dim,
                    kv_owners,
                    ..
                } = self.plan.layout().attention(layer)?
                else {
                    continue;
                };
                let local = self
                    .plan
                    .layout()
                    .state_schema()
                    .entry(layer)
                    .and_then(|e| e.local_kv_layer)
                    .ok_or_else(|| error("trace local KV layer absent"))?;
                if !trace.borrow().accepts_metadata(kv_owners.len(), 0) {
                    return Err(error("committed trace metadata limit"));
                }
                pool.export_committed_bf16_kv(&local_source,local,words,|chunk| {
                    if !trace.borrow_mut().accepts("committed.kv",chunk.words.len()*4) {
                        return Err(error("committed trace capture limit/queue pressure"));
                    }
                    let mut data = Vec::with_capacity(chunk.words.len() * 4);
                    for word in chunk.words {
                        data.extend_from_slice(&f32::from_bits((*word as u32) << 16).to_le_bytes());
                    }
                    trace.borrow_mut().record(serde_json::json!({"event":"committed.kv","committed":true,
                        "transaction":transaction.get(),"layer":layer,"local_kv_layer":local,"plane":chunk.plane,
                        "positions":[chunk.positions.start,chunk.positions.end],"shape":[chunk.positions.len(),kv_heads.len(),head_dim],
                        "strides":[kv_heads.len()*head_dim,head_dim,1],"native_dtype":"Bf16",
                        "kv_heads":[kv_heads.start,kv_heads.end],"kv_owners":kv_owners.iter().map(|r|r.get()).collect::<Vec<_>>(),
                        "page":chunk.page.0,"slot":chunk.slot,"slot_epoch":chunk.slot_epoch,
                        "source_owner":source.owner,"state_slot":source.state_slot.get(),"generation":source.generation,"incarnation":source.incarnation}),data);
                    if trace.borrow().failed() {
                        return Err(error("committed trace writer incomplete"));
                    }
                    Ok(())
                })?;
            }
            Ok(())
        })();
        trace.borrow_mut().committed_end();
        self.completion(result)
    }
    fn shutdown(&mut self) -> Result<()> {
        if self.closed {
            return self.completion(Ok(()));
        }
        if let Err(error) = self.poll_quiescence() {
            emit_cache_snapshot(
                self.plan.layout().owner().get(),
                self.device.device_ordinal(),
                "shutdown_unproven",
                self.operators
                    .as_ref()
                    .and_then(|ops| ops.mixed_decoded_cache_snapshot()),
            );
            return Err(error);
        }
        if let Some(trace) = self.trace.take() {
            if let Some(ops) = &self.operators {
                ops.clear_diagnostic_trace();
            }
            trace.borrow_mut().close();
        }
        // Keep the image alive through worker state/KV retirement, just like the
        // F32 leaf. Its endpoint closes when worker custody is finally dropped.
        self.closed = true;
        let result = self.completion(Ok(()));
        emit_cache_snapshot(
            self.plan.layout().owner().get(),
            self.device.device_ordinal(),
            "shutdown",
            self.operators
                .as_ref()
                .and_then(|ops| ops.mixed_decoded_cache_snapshot()),
        );
        tracing::info!(
            owner = self.plan.layout().owner().get(),
            device = self.device.device_ordinal(),
            quiescent = result.is_ok(),
            device_quarantine = self.device.needs_quarantine(),
            operator_quarantine = self.operators.as_ref().is_some_and(|o| o.needs_quarantine()),
            allocator = ?self.device.operators().allocator_metrics(),
            "Mixed owner shutdown fence; image retained until worker retirement"
        );
        result
    }
}
impl Drop for MixedCudaTensorProgram {
    fn drop(&mut self) {
        if std::thread::panicking()
            || self.device.needs_quarantine()
            || self
                .operators
                .as_ref()
                .is_some_and(|o| o.needs_quarantine())
        {
            std::mem::forget(self.operators.take());
            std::mem::forget(self.lifecycle.take());
            std::mem::forget(self.device.clone());
        }
    }
}
impl HybridCudaTensorStage<MixedCudaTensorProgram> {
    /// Owner-local typed leaf for the existing generic tensor constructor.
    /// The sealed program fixes full local residency, capacities and route/data sites.
    pub fn prepare_mixed(
        resources: &BoundDecoderResources,
        inputs: &MixedTensorExecutionPlan,
        lane: usize,
        device: HybridCudaDevice,
        endpoint: DecoderTensorCollective,
    ) -> Result<Self> {
        Self::prepare_mixed_with_trace(resources, inputs, lane, device, endpoint, None)
    }
    #[doc(hidden)]
    pub fn prepare_mixed_with_trace(
        resources: &BoundDecoderResources,
        inputs: &MixedTensorExecutionPlan,
        lane: usize,
        device: HybridCudaDevice,
        endpoint: DecoderTensorCollective,
        trace_config: Option<MixedTraceConfig>,
    ) -> Result<Self> {
        let config = inputs.config();
        let plan = inputs.program(lane)?.clone();
        plan.validate_spec(resources.spec())?;
        if !resources.state_dict().validate_source_identities()
            || std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() != Ok(":0:0")
        {
            return Err(error(
                "Mixed startup requires fresh source and launcher CUBLAS_WORKSPACE_CONFIG=:0:0",
            ));
        }
        let owner = plan.layout().owner();
        let endpoint = MixedRankCollective::prepare(endpoint, owner, inputs.collective().clone())?;
        plan.validate_collective(&endpoint)?;
        let control = endpoint.control();
        let description = description(
            resources.spec(),
            plan.segment().clone(),
            plan.layout().state_schema(),
            config,
            None,
        )?;
        let capabilities = description.execution_capabilities()?;
        let planes = plan.layout().state_schema().kv_planes_with_dtype(
            config.page_size,
            config.max_positions,
            config.dtype(),
        )?;
        let (estimate, context) = plan.cuda_claim_inputs(
            resources,
            description.physical_pages,
            device.device_ordinal(),
            config.max_parameter_bytes,
            inputs.attention_shapes(),
        )?;
        context.validate_device_budget(device.device_ordinal(), device.budget())?;
        if let Some(trace) = &trace_config {
            trace.validate_admission(
                device.budget().workspace_bytes,
                estimate.workspace_bytes_upper_bound,
                description.physical_pages,
            )?;
        }
        let claim = cuda_call(|| device.admit_with_context(estimate, context))?;
        let mut ready_cache_snapshot = None;
        let result = cuda_call(|| {
            let mut operators = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
                Rc::clone(device.operators()),
                resources.state_dict().parameters(),
                ExpertCachePolicy::KeepAll,
                0,
                NumericFp8Precision::W8A16Bf16F32Accumulate,
            )?;
            // Subscriber policy is explicit; setting a log filter alone must not
            // leave the model's independent first profiling switch disabled.
            operators.set_mixed_rank_performance(
                tracing::enabled!(target: "ferrule_performance", tracing::Level::DEBUG),
            )?;
            let lifecycle = operators.prepare_mixed_rank_program_with_claim(
                resources,
                plan.clone(),
                device.clone(),
                config.max_parameter_bytes,
                Box::new(endpoint),
                inputs.attention_shapes(),
                &claim,
            )?;
            let trace = trace_config
                .map(|config| MixedTraceWriter::new(config, owner.get()))
                .transpose()?
                .map(|writer| Rc::new(RefCell::new(writer)));
            if let Some(trace) = &trace {
                let mut binding = serde_json::json!({
                    "owner":owner.get(),"device":device.device_ordinal(),
                    "epoch":inputs.collective().epoch().get(),"group":inputs.collective().group().get(),
                    "source":inputs.collective().source().get(),
                    "ordered_members":inputs.collective().members().iter().map(|r|r.get()).collect::<Vec<_>>(),
                    "data_sites":bounded_debug(&inputs.collective().sites())?,
                    "route_sites":bounded_debug(&inputs.collective().route_sites())?,
                    "rank_layout":{
                        "lane":plan.layout().lane(),"layers":[plan.layers().start,plan.layers().end],
                        "precision":bounded_debug(&plan.layout().precision())?,
                        "routed_execution":bounded_debug(&plan.layout().routed_execution())?
                    }
                });
                if let Some(b) = plan.decoded_weight_cache_budget() {
                    binding["decoded_weight_cache"] = serde_json::json!({
                        "capacity_grants":b.capacity_grants,"pinned_grants":b.pinned_grants,
                        "routed_working_set_grants":b.routed_working_set_grants,"routed_entries":b.routed_entries,
                    });
                }
                trace.borrow_mut().seal(binding)?;
                let limits = trace.borrow().config().limits()?;
                let filter = Rc::clone(trace);
                let sink = Rc::clone(trace);
                operators.set_native_diagnostic_trace(limits,
                    move |descriptor| filter.borrow_mut().accepts_descriptor(descriptor),
                    move |event| {
                        use ferrule_model::transformer::CudaDiagnosticBuffer;
                        let d=event.descriptor;
                        // Router sink selection precedes the kernel; another native
                        // event may have consumed queue credit before this callback.
                        if !sink.borrow_mut().accepts_descriptor(d) { return Ok(()); }
                        if !sink.borrow().accepts_metadata(
                            event.routes.map_or(0, |r| r.expert_ids().len().saturating_add(r.weights().len())),
                            event.parameter.map_or(0, |p| p.path().as_str().len()),
                        ) { return Ok(()); }
                        let values=match event.buffer {
                            CudaDiagnosticBuffer::F32(buffer)=>event.operators.download_f32_buffer(buffer)?,
                            CudaDiagnosticBuffer::Bf16(buffer)=>event.operators.download_bf16_buffer(buffer)?,
                        };
                        let metadata=serde_json::json!({
                            "event":d.name,"layer":d.layer,"shape":d.layout.dimensions,"strides":d.layout.strides,
                            "native_dtype":format!("{:?}",d.layout.dtype),"committed":false,
                            "sequence":d.sequence.as_ref().map(|s|serde_json::json!({"sequence":s.sequence,"packed_rows":s.packed_rows,"positions":s.positions,"input_frontier":s.input_frontier})),
                            "parameter_partition":d.parameter_partition.as_ref().map(|p|serde_json::json!({"rows":p.rows.iter().map(|r|[r.start,r.end]).collect::<Vec<_>>(),"columns":[p.columns.start,p.columns.end],"collective":format!("{:?}",p.collective)})),
                            "attention_partition":d.attention_partition.as_ref().map(attention_partition),
                            "parameter":event.parameter.map(|p|p.path().to_string()),
                            "role":event.parameter.map(|p|format!("{:?}",p.role())),
                            "ids":event.routes.map(|r|r.expert_ids()),
                            "weight_bits":event.routes.map(|r|r.weights().iter().map(|v|v.to_bits()).collect::<Vec<_>>())
                        });
                        let mut data = Vec::with_capacity(values.len() * 4);
                        for value in values { data.extend_from_slice(&value.to_le_bytes()); }
                        sink.borrow_mut().record(metadata, data);
                        Ok(())
                    });
            }
            let mut program = MixedCudaTensorProgram {
                trace,
                plan,
                operators: Some(operators),
                lifecycle: Some(lifecycle),
                device: device.clone(),
                hidden: resources.spec().hidden_size(),
                failed: false,
                closed: false,
            };
            let prepared = (|| {
                for layer in program.plan.layers() {
                    if matches!(
                        resources.spec().layers()[layer].feed_forward(),
                        ferrule_model::transformer::FeedForward::Moe(_)
                    ) && !program
                        .operators
                        .as_ref()
                        .expect("prepared operators")
                        .mixed_rank_all_expert_shards_resident(layer)
                    {
                        return Err(error("Mixed owner missing resident local expert shards"));
                    }
                }
                let pool = TypedCudaPagedKvPool::from_strategy(
                    Rc::clone(device.operators()),
                    &planes,
                    description.physical_pages,
                )?;
                // Verify initial state allocation under the same admitted credits.
                // Persistent sessions are subsequently created by the worker journal.
                let life = program.lifecycle.as_mut().expect("prepared lifecycle");
                let state = life.create()?;
                if let Err(failure) = life.try_release(state) {
                    let (error, state) = failure.into_parts();
                    std::mem::forget(state);
                    return Err(CompletionUnknown::wrap(error));
                }
                Ok(pool)
            })();
            let pool = match prepared {
                Ok(pool) => pool,
                Err(primary) => {
                    let cleanup = program.shutdown();
                    return program.completion(Err(Error::with_cleanup(
                        "Mixed pool/state preparation",
                        primary,
                        cleanup,
                    )));
                }
            };
            tracing::info!(
                owner = owner.get(),
                device = device.device_ordinal(),
                live_grants = device.operators().allocator_metrics().live_granted_bytes,
                ?estimate,
                "Mixed owner image/KV ready; claim final fence pending"
            );
            ready_cache_snapshot = program
                .operators
                .as_ref()
                .and_then(|ops| ops.mixed_decoded_cache_snapshot());
            if program.plan.decoded_weight_cache_budget().is_some()
                && ready_cache_snapshot.is_none()
            {
                let cleanup = program.shutdown();
                return program.completion(Err(Error::with_cleanup(
                    "cache-on Mixed preparation did not publish decoded cache custody",
                    error("missing decoded cache snapshot"),
                    cleanup,
                )));
            }
            Self::new(
                TensorScopedStateProgram::new(program, control, owner),
                PagedKvBackend::new(pool),
                description,
                capabilities,
            )
        });
        let result = checked(&device, claim.finish(result));
        if result.is_ok() {
            // No cache mutation occurs between the prepared snapshot and this
            // successful claim fence; never label a failed claim as ready.
            emit_cache_snapshot(
                owner.get(),
                device.device_ordinal(),
                "ready",
                ready_cache_snapshot,
            );
        }
        tracing::info!(
            owner = owner.get(),
            device = device.device_ordinal(),
            claim_finished = result.is_ok(),
            device_quarantine = device.needs_quarantine(),
            allocator = ?device.operators().allocator_metrics(),
            "Mixed owner preparation claim completion"
        );
        result
    }
}
