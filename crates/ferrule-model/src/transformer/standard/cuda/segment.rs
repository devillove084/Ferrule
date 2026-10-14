//! Owner-local CUDA adapter over the shared stateful segment. No CUDA state is
//! cloned or sent between threads; only explicit activation boundaries travel.
use super::hybrid::external_experts::{ExternalExpertCustody, RoutedCall};
use super::{CudaStandardDecoderOperators, CudaStandardKvBinding};
use crate::decoder::{CudaHybridSequenceLifecycle, GenericDecoderOptions, HybridCudaRoutedExperts};
use crate::decoder::{
    CudaHybridSequenceState, CudaKvView, HybridCudaDevice, HybridStateSchema, PackedDecoderBatch,
};
use crate::execution::ExecutionPrecisionPolicy;
use crate::transformer::{
    Attention, BoundDecoderResources, BoundParameter, DeviceSegmentInput, DeviceSegmentOutput,
    FeedForward, HybridDecoderSegment, LayerSegmentPlan, SegmentError, SegmentInput, SegmentOutput,
    SegmentResult, SegmentStage,
};
use crate::transformer::{ExpertCachePolicy, NumericFp8Precision};
use ferrule_common::{Error, ParallelRankId};
use ferrule_common::{Result, execution::ExecutionTransactionId};

/// The image must be constructed inside its owner thread, never moved there
/// after device preparation.
/// ```compile_fail
/// use ferrule_model::transformer::CudaHybridDecoderSegment;
/// fn require_send<T: Send>() {}
/// require_send::<CudaHybridDecoderSegment>();
/// ```
#[derive(Debug)]
pub struct CudaHybridDecoderSegment {
    segment: HybridDecoderSegment,
    operators: CudaStandardDecoderOperators,
    device: HybridCudaDevice,
    external: ExternalExpertCustody,
    failed: bool,
    closed: bool,
    shutdown_started: bool,
    watermark: Option<ExecutionTransactionId>,
}
impl CudaHybridDecoderSegment {
    pub fn prepare(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        precision: ExecutionPrecisionPolicy,
        max_positions: usize,
        max_parameter_bytes: u64,
        device: HybridCudaDevice,
    ) -> SegmentResult<Self> {
        Self::prepare_inner(
            resources,
            plan,
            precision,
            max_positions,
            max_parameter_bytes,
            device,
            None,
            None,
        )
    }

    /// Metadata-only gate for CUDA TP1 with mandatory external routed experts.
    /// No checkpoint payload is read and no CUDA context is required. Runtime
    /// must separately admit the stage root, physical KV/state, transport holds
    /// and every external child, aggregating colocated owners on each device.
    pub fn validate_routed_preparation(
        resources: &BoundDecoderResources,
        plan: &LayerSegmentPlan,
        options: &GenericDecoderOptions,
    ) -> Result<()> {
        options.validate_hybrid_cuda(resources)?;
        if plan.total_layers() != resources.spec().layers().len()
            || options
                .active_layers()
                .is_some_and(|n| n != plan.total_layers())
        {
            return Err(super::cuda_error(
                "external CUDA segment requires a full-model plan, not an active prefix",
            ));
        }
        if !matches!(
            options.hybrid_cuda_numeric_fp8_precision(),
            Some(NumericFp8Precision::F32Tf32x3 | NumericFp8Precision::W8A16Bf16F32Accumulate)
        ) {
            return Err(super::cuda_error(
                "external CUDA segment requires explicit numeric F32 or W8A16; no precision fallback",
            ));
        }
        let layers = &resources.spec().layers()[plan.layers()];
        if !layers
            .iter()
            .any(|layer| matches!(layer.feed_forward(), FeedForward::Moe(_)))
        {
            return Err(super::cuda_error(
                "external CUDA segment has no routed layer",
            ));
        }
        HybridStateSchema::for_segment(resources.spec(), plan.layers())?;
        for layer in layers {
            if let Attention::Gqa(gqa) = layer.attention() {
                super::validate_rope(gqa.rotary())?;
            }
        }
        Ok(())
    }

    /// Construct on the persistent root owner. Numeric arithmetic is taken
    /// verbatim from `options`; the legacy dense constructor is unchanged.
    /// All selected non-expert bindings are resident before return. Routed
    /// weights are never loaded on this root and cannot fall back to local EP.
    ///
    /// This interface does NOT enable TP: it uses full-width TP1 descriptors.
    /// The attachment must use this exact device's operators for GPU combine.
    pub fn prepare_with_routed_experts(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        options: &GenericDecoderOptions,
        device: HybridCudaDevice,
        experts: HybridCudaRoutedExperts,
    ) -> SegmentResult<Self> {
        Self::prepare_inner(
            resources,
            plan,
            options.precision(),
            options.max_positions(),
            options.max_parameter_bytes(),
            device,
            Some(options),
            Some(experts),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_inner(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        precision: ExecutionPrecisionPolicy,
        max_positions: usize,
        max_parameter_bytes: u64,
        device: HybridCudaDevice,
        options: Option<&GenericDecoderOptions>,
        experts: Option<HybridCudaRoutedExperts>,
    ) -> SegmentResult<Self> {
        let prepare = |source| SegmentError::Execution {
            stage: SegmentStage::Prepare,
            source,
        };
        let mut external = ExternalExpertCustody::new(experts, &device);
        let mut claim = None;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device.ensure_ready().map_err(prepare)?;
            if let Some(options) = options {
                Self::validate_routed_preparation(resources, &plan, options).map_err(prepare)?;
                let (_, scratch) = options
                    .hybrid_cuda_numeric_fp8()
                    .expect("validated numeric profile");
                if scratch > device.budget().workspace_bytes {
                    return Err(prepare(super::cuda_error(
                        "numeric reservation exceeds root workspace budget",
                    )));
                }
            } else if let Some(layers) = resources.spec().layers().get(plan.layers()) {
                for layer in layers {
                    if !matches!(layer.feed_forward(), FeedForward::SwiGlu(_)) {
                        return Err(prepare(super::cuda_error(
                            "hybrid CUDA segment requires external EP for routed layers",
                        )));
                    }
                    if let Attention::Gqa(gqa) = layer.attention() {
                        super::validate_rope(gqa.rotary()).map_err(prepare)?;
                    }
                }
            }
            // PP admission remains the caller's responsibility. Reserve its
            // already-admitted root ceilings under the same device/image claim
            // as the full module and TP rank, before any parameter upload.
            let budget = device.budget();
            claim = Some(
                device
                    .admit(
                        crate::decoder::HybridCudaMemoryEstimate {
                            per_sequence_state_bytes: 0,
                            kv_bytes: 0,
                            weight_bytes_upper_bound: budget.weight_bytes,
                            workspace_bytes_upper_bound: budget.workspace_bytes,
                        },
                        0,
                    )
                    .map_err(prepare)?,
            );
            let authority = claim.as_ref().expect("admitted segment");
            authority.begin_image(&device).map_err(prepare)?;
            let segment = HybridDecoderSegment::prepare(
                resources,
                plan,
                precision,
                max_positions,
                max_parameter_bytes,
            )?;
            let mut operators =
                match options.and_then(GenericDecoderOptions::hybrid_cuda_numeric_fp8_config) {
                    Some((limits, scratch, arithmetic)) => {
                        CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
                            std::rc::Rc::clone(device.operators()),
                            segment.parameters(),
                            ExpertCachePolicy::Bounded(limits),
                            scratch,
                            arithmetic,
                        )
                    }
                    None => CudaStandardDecoderOperators::new(
                        std::rc::Rc::clone(device.operators()),
                        precision,
                        segment.parameters(),
                    ),
                }
                .map_err(prepare)?;
            let result = segment.prepare_cuda_bindings(&mut operators);
            if operators.needs_quarantine() {
                device.quarantine();
            }
            result.map_err(prepare)?;
            if operators.resident_parameter_bytes() > device.budget().weight_bytes {
                return Err(prepare(super::cuda_error(
                    "CUDA segment weights exceed admitted budget",
                )));
            }
            authority.image_prepared().map_err(prepare)?;
            Ok((segment, operators))
        }));
        let result = match result {
            Ok(Ok((segment, operators))) => Ok(Self {
                segment,
                operators,
                device,
                external,
                failed: false,
                closed: false,
                shutdown_started: false,
                watermark: None,
            }),
            Ok(Err(primary)) => {
                // No program handle can be returned on failed preparation. A
                // pending/failed cleanup therefore retains and quarantines custody.
                let cleanup = external.require_drain(&device, true);
                if cleanup.is_ok() {
                    external.close();
                } else {
                    device.quarantine();
                    external.retain_unknown();
                    std::mem::forget(device.clone());
                }
                drop(external);
                match cleanup {
                    Ok(()) => Err(primary),
                    Err(cleanup) => Err(prepare(Error::with_cleanup(
                        "CUDA segment expert startup",
                        Error::ModelSource {
                            source: Box::new(primary),
                        },
                        Err(cleanup),
                    ))),
                }
            }
            Err(panic) => {
                device.quarantine();
                external.retain_unknown();
                std::mem::forget(device.clone());
                std::panic::resume_unwind(panic)
            }
        };
        match claim {
            Some(claim) => match result {
                Ok(image) => claim.finish(Ok(image)).map_err(prepare),
                Err(primary) => match claim.rollback_after_failure() {
                    Ok(()) => Err(primary),
                    Err(cleanup) => Err(prepare(Error::with_cleanup(
                        "CUDA segment preparation",
                        Error::ModelSource {
                            source: Box::new(primary),
                        },
                        Err(cleanup),
                    ))),
                },
            },
            None => result,
        }
    }

    pub fn precision(&self) -> ExecutionPrecisionPolicy {
        ExecutionPrecisionPolicy::f32()
    }
    pub fn activation_dtype(&self) -> crate::transformer::RowsDType {
        crate::transformer::RowsDType::F32
    }
    pub fn kv_element_type(&self) -> ferrule_common::execution::KvElementType {
        ferrule_common::execution::KvElementType::F32
    }
    pub fn plan(&self) -> &LayerSegmentPlan {
        self.segment.plan()
    }
    pub fn schema(&self) -> &HybridStateSchema {
        self.segment.schema()
    }
    pub fn parameters(&self) -> &[BoundParameter] {
        self.segment.parameters()
    }
    pub fn operators(&self) -> &CudaStandardDecoderOperators {
        &self.operators
    }
    pub fn resident_parameter_bytes(&self) -> usize {
        self.operators.resident_parameter_bytes()
    }
    pub fn needs_quarantine(&self) -> bool {
        self.device.needs_quarantine() || self.operators.needs_quarantine()
    }
    pub fn validate_execution(
        &self,
        batch: &PackedDecoderBatch,
        states: &[CudaHybridSequenceState],
    ) -> SegmentResult<()> {
        self.segment.validate_execution(batch, states)
    }
    pub fn validate_frontier(
        &self,
        batch: &PackedDecoderBatch,
        states: &[CudaHybridSequenceState],
    ) -> SegmentResult<()> {
        self.segment.validate_frontier(batch, states)
    }
    /// A lifecycle over this stage's compact schema and exact CUDA owner.
    pub fn new_lifecycle(&self) -> Result<CudaHybridSequenceLifecycle> {
        self.device.ensure_ready()?;
        if self.closed || self.shutdown_started {
            return Err(super::cuda_error("CUDA segment is shutting down"));
        }
        CudaHybridSequenceLifecycle::new(self.schema().clone(), self.device.clone())
    }
    pub fn routed_source_rank(&self) -> Option<ParallelRankId> {
        self.external
            .attachment
            .as_ref()
            .map(|experts| experts.source_rank)
    }
    pub fn numeric_fp8_precision(&self) -> Option<NumericFp8Precision> {
        self.operators.numeric_fp8_precision()
    }
    pub fn retains_expert_custody(&self) -> bool {
        self.external.attachment.as_ref().is_some_and(|a| !a.closed)
            || self.external.forward.is_some()
            || !self.external.holds.is_empty()
            || self.external.kv.is_some()
    }
    fn root_completion(&mut self) -> Result<()> {
        let result = self.operators.quiesce();
        if self.operators.needs_quarantine() {
            self.device.quarantine();
        }
        result?;
        self.device.fence()
    }
    /// Poll cancellation/quiescence without replaying forward. Pending or Err
    /// retains KV/activation custody. Only true permits the caller's existing
    /// journal to roll back/discard working state; it does not publish anything.
    pub fn poll_quiescence(&mut self) -> Result<bool> {
        self.external.begin_drain()?;
        if !self.external.drain(&self.device, false)? {
            return Ok(false);
        }
        self.root_completion()?;
        self.external.finish();
        self.failed = false;
        Ok(true)
    }
    pub fn quiesce(&mut self) -> Result<()> {
        if !self.poll_quiescence()? {
            return Err(super::cuda_error(
                "external expert completion pending; retain segment/KV custody",
            ));
        }
        Ok(())
    }
    /// Close child owners before releasing the image. A pending shutdown may be
    /// polled again, but a new forward is permanently rejected once this starts.
    pub fn poll_shutdown(&mut self) -> Result<bool> {
        self.shutdown_started = true;
        if self.closed {
            self.device.ensure_ready()?;
            return Ok(true);
        }
        self.external.begin_drain()?;
        if !self.external.drain(&self.device, true)? {
            return Ok(false);
        }
        self.root_completion()?;
        self.external.close();
        self.closed = true;
        Ok(true)
    }
    pub fn shutdown(&mut self) -> Result<()> {
        if !self.poll_shutdown()? {
            return Err(super::cuda_error(
                "external expert shutdown pending; retain segment/KV custody",
            ));
        }
        Ok(())
    }
    pub fn execute_cuda_bound(
        &mut self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        states: &mut [CudaHybridSequenceState],
        kv: &mut CudaKvView,
    ) -> SegmentResult<DeviceSegmentOutput> {
        self.execute_cuda_bound_checked(batch, input, states, kv, &mut |_| Ok(()))
    }
    pub fn execute_cuda_bound_checked(
        &mut self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        states: &mut [CudaHybridSequenceState],
        kv: &mut CudaKvView,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> SegmentResult<DeviceSegmentOutput> {
        if self.external.attachment.is_some() {
            return Err(SegmentError::Execution {
                stage: SegmentStage::Input,
                source: super::cuda_error(
                    "external CUDA segment requires explicit transaction/source/session binding",
                ),
            });
        }
        self.execute_inner(batch, input, states, kv, None, check_active)
    }

    /// Execute external EP on the one shared layer loop. Sequence IDs are in
    /// packed-sequence order (not state-slot or token-row order), supplied by the
    /// authenticated runtime scope. This call never commits logical state or publishes KV.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cuda_bound_with_routed_experts(
        &mut self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        states: &mut [CudaHybridSequenceState],
        kv: &mut CudaKvView,
        context: (ExecutionTransactionId, ParallelRankId, &[u64]),
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> SegmentResult<DeviceSegmentOutput> {
        self.execute_inner(batch, input, states, kv, Some(context), check_active)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_inner(
        &mut self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        states: &mut [CudaHybridSequenceState],
        kv: &mut CudaKvView,
        context: Option<(ExecutionTransactionId, ParallelRankId, &[u64])>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> SegmentResult<DeviceSegmentOutput> {
        let input_error = |source| SegmentError::Execution {
            stage: SegmentStage::Input,
            source,
        };
        self.device.ensure_ready().map_err(input_error)?;
        if self.closed
            || self.shutdown_started
            || self.failed
            || self.external.forward.is_some()
            || self.watermark.is_some_and(|last| last >= kv.transaction())
        {
            return Err(input_error(super::cuda_error(
                "CUDA segment is closed, stale, or awaiting rollback proof",
            )));
        }
        match (self.routed_source_rank(), context) {
            (None, None) => {}
            (Some(source), Some((transaction, rank, sequences)))
                if transaction == kv.transaction()
                    && rank == source
                    && sequences.len() == batch.sequences().len() => {}
            _ => {
                return Err(input_error(super::cuda_error(
                    "external CUDA segment transaction/source/session binding mismatch",
                )));
            }
        }
        check_active(kv.transaction()).map_err(input_error)?;

        self.validate_execution(batch, states)?;
        // Validate ALL recurrent owners before embedding or any KV write.
        for sequence in batch.sequences() {
            for state in states[sequence.state_index()]
                .kv_state()
                .layers()
                .iter()
                .flatten()
            {
                state
                    .validate_owner(self.device.operators())
                    .map_err(input_error)?;
            }
        }
        // Validate KV owner and all compact planes before any recurrent mutation.
        for layer in self.schema().layers() {
            if let crate::decoder::HybridLayerSchema::FullAttention {
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
                        if planes.layout.elements_per_token != kv_heads * head_dim {
                            return Err(super::cuda_error("hybrid segment KV geometry mismatch"));
                        }
                        Ok(())
                    },
                )
                .map_err(input_error)?;
            }
        }
        self.watermark = Some(kv.transaction());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.external
                .begin(kv.transaction(), batch, kv)
                .map_err(input_error)?;
            self.operators
                .preflight_expert_metadata()
                .map_err(input_error)?;
            let result = {
                let mut binding = CudaStandardKvBinding::new(kv, batch).map_err(input_error)?;
                if let Some((transaction, source, sequences)) = context {
                    let forward = self
                        .external
                        .forward
                        .as_ref()
                        .expect("begun external forward");
                    let experts = self
                        .external
                        .attachment
                        .as_mut()
                        .expect("validated attachment");
                    let mut active = |id| {
                        self.device.ensure_ready()?;
                        forward.check(id, batch)?;
                        check_active(id)?;
                        if let Some(check) = &mut experts.check_active {
                            check(id)?;
                        }
                        Ok(())
                    };
                    let mut call = RoutedCall {
                        executor: experts
                            .executor
                            .as_mut()
                            .expect("attached executor")
                            .as_mut(),
                        device: &self.device,
                        holds: &mut self.external.holds,
                    };
                    self.segment.execute_bound_inner(
                        batch,
                        input,
                        states,
                        &mut binding,
                        super::super::LayerOperators::Cuda(&mut self.operators),
                        &mut active,
                        Some(((transaction, source, sequences), &mut call)),
                    )
                } else {
                    self.segment.execute_bound(
                        batch,
                        input,
                        states,
                        &mut binding,
                        super::super::LayerOperators::Cuda(&mut self.operators),
                        check_active,
                    )
                }
            };
            if self.operators.needs_quarantine() {
                self.device.quarantine();
            }
            let output = result?;
            let completion = (|| {
                self.external.require_drain(&self.device, false)?;
                self.root_completion()?;
                self.external.check(&self.device, kv.transaction(), batch)?;
                check_active(kv.transaction())
            })();
            if let Err(source) = completion {
                if self.needs_quarantine() {
                    std::mem::forget(output);
                }
                return Err(SegmentError::Execution {
                    stage: SegmentStage::Output,
                    source,
                });
            }
            self.external.finish();
            Ok(output)
        }));
        if self.operators.needs_quarantine() {
            self.device.quarantine();
        }
        match result {
            Ok(result) => result.map_err(|primary| {
                self.failed = true;
                match self.external.fail() {
                    Ok(()) => primary,
                    Err(cleanup) => SegmentError::Execution {
                        stage: SegmentStage::Output,
                        source: Error::with_cleanup(
                            "CUDA segment expert cancellation",
                            Error::ModelSource {
                                source: Box::new(primary),
                            },
                            Err(cleanup),
                        ),
                    },
                }
            }),
            Err(panic) => {
                self.device.quarantine();
                self.operators.poisoned = true;
                std::panic::resume_unwind(panic)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_routed_experts(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [CudaHybridSequenceState],
        kv: &mut CudaKvView,
        context: (ExecutionTransactionId, ParallelRankId, &[u64]),
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> SegmentResult<SegmentOutput> {
        let output = self.execute_cuda_bound_with_routed_experts(
            batch,
            input.into(),
            states,
            kv,
            context,
            check_active,
        )?;
        self.finish_host_output(output, kv.transaction(), check_active)
    }
    pub fn execute(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [CudaHybridSequenceState],
        kv: &mut CudaKvView,
    ) -> SegmentResult<SegmentOutput> {
        self.execute_checked(batch, input, states, kv, &mut |_| Ok(()))
    }
    pub fn execute_checked(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [CudaHybridSequenceState],
        kv: &mut CudaKvView,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> SegmentResult<SegmentOutput> {
        let output =
            self.execute_cuda_bound_checked(batch, input.into(), states, kv, check_active)?;
        self.finish_host_output(output, kv.transaction(), check_active)
    }
    fn finish_host_output(
        &mut self,
        output: DeviceSegmentOutput,
        transaction: ExecutionTransactionId,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> SegmentResult<SegmentOutput> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let output = self.download_output(output)?;
            check_active(transaction).map_err(|source| SegmentError::Execution {
                stage: SegmentStage::Output,
                source,
            })?;
            Ok(output)
        }));
        match result {
            Ok(result) => {
                self.failed |= result.is_err();
                result
            }
            Err(panic) => {
                self.device.quarantine();
                self.operators.poisoned = true;
                std::panic::resume_unwind(panic)
            }
        }
    }
    pub fn download_output(&mut self, output: DeviceSegmentOutput) -> SegmentResult<SegmentOutput> {
        let result = output.into_host(&mut self.operators);
        if self.operators.needs_quarantine() {
            self.device.quarantine();
        }
        result
    }
}
impl Drop for CudaHybridDecoderSegment {
    fn drop(&mut self) {
        let complete = !std::thread::panicking()
            && matches!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.shutdown())),
                Ok(Ok(()))
            );
        if !complete || self.needs_quarantine() {
            self.device.quarantine();
            self.operators.poisoned = true;
            self.external.retain_unknown();
            // Operator Drop retains buffers; keep their owner and budget alive too.
            std::mem::forget(self.device.clone());
        }
    }
}
