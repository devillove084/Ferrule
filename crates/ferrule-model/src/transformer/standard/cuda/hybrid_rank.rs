//! CUDA TP1/2/4 owner over the shared hybrid segment loop. Host staging is only
//! communication; all projections, attention, recurrence and FFNs use CUDA.
use super::hybrid::external_experts::{ExternalExpertCustody, RoutedCall};
use super::*;
use crate::TensorRole;
use crate::decoder::{
    CudaHybridSequenceLifecycle, CudaHybridSequenceState, CudaKvView, GenericDecoderOptions,
    HybridCudaDevice, HybridCudaRoutedExperts, HybridLayerSchema, HybridStateSchema,
    PackedDecoderBatch,
};
use crate::nn::{ParameterId, ParameterResidency};
use crate::transformer::expert_parallel::{RoutedSwiGluExecutor, RoutedSwiGluRequest};
use crate::transformer::hybrid_parallel::*;
use crate::transformer::parallel::{TensorParallelCollective, TensorParallelStagePlan};
use crate::transformer::{
    BoundDecoderResources, CudaNumericFp8RankBinding, DeviceSegmentInput, DeviceSegmentOutput,
    HybridDecoderSegment, HybridRankStorage, LayerSegmentPlan, NumericFp8RankPlan, SegmentInput,
    SegmentOutput, StateDictMaterializer,
};
use std::cell::RefCell;

mod channel;
mod lifecycle;
mod memory;
mod mixed;
mod mixed_layer;
mod operators;
use channel::RankChannel;
pub use lifecycle::{CudaHybridRankLifecycle, CudaHybridRankState};
pub use memory::CudaHybridRankMemory;
pub use mixed::CudaMixedRankLinear;
pub(super) use mixed::MixedRankImage;
pub(super) use mixed_layer::MixedGqaInvocation;

pub(super) struct RankOperators {
    pub layout: Arc<HybridRankLayout>,
    channel: Rc<RefCell<RankChannel>>,
    max_rows: usize,
    linears: BTreeMap<(ParameterId, TensorRole), Rc<RankLinear>>,
}
struct RankLinear {
    binding: BoundParameter,
    shape: [usize; 2],
    partition: HybridParameterPartition,
    storage: RankLinearStorage,
}
enum RankLinearStorage {
    Numeric(CudaNumericFp8RankBinding),
    Dense(ferrule_backend::cuda::operators::linear::CudaArtifactLinearHandle),
}

pub struct CudaHybridRankSegment {
    segment: HybridDecoderSegment,
    parallel: HybridParallelPlan,
    layout: Arc<HybridRankLayout>,
    channel: Rc<RefCell<RankChannel>>,
    operators: CudaStandardDecoderOperators,
    device: HybridCudaDevice,
    external: ExternalExpertCustody,
    state_credits: Rc<lifecycle::StateCredits>,
    memory: CudaHybridRankMemory,
    expert_boundary: Boundary,
    watermark: Option<ExecutionTransactionId>,
    failed: bool,
    closed: bool,
    shutdown_started: bool,
}
impl std::fmt::Debug for CudaHybridRankSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaHybridRankSegment")
            .field("layout", &self.layout)
            .field("memory", &self.memory)
            .field("failed", &self.failed)
            .finish_non_exhaustive()
    }
}
fn segment_error(error: crate::transformer::SegmentError) -> Error {
    Error::ModelSource {
        source: Box::new(error),
    }
}
impl CudaHybridRankSegment {
    /// Build inside the rank's persistent CUDA owner. `parallel` supplies the
    /// model-owned V2 schedule; `layout` must be exactly its selected lane.
    /// EP custody is attached on the designated source only, never on peers.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_rank(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        parallel: HybridParallelPlan,
        layout: Arc<HybridRankLayout>,
        options: &GenericDecoderOptions,
        device: HybridCudaDevice,
        collective: Box<dyn StandardTensorCollective>,
        experts: Option<HybridCudaRoutedExperts>,
    ) -> Result<Self> {
        Self::prepare_rank_inner(
            resources, plan, parallel, layout, options, device, collective, experts, None,
        )
    }

    /// Use the factory's opaque admission token. The caller must finish it only
    /// after all pool/lifecycle/initial-state construction has succeeded.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_rank_with_claim(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        parallel: HybridParallelPlan,
        layout: Arc<HybridRankLayout>,
        options: &GenericDecoderOptions,
        device: HybridCudaDevice,
        collective: Box<dyn StandardTensorCollective>,
        experts: Option<HybridCudaRoutedExperts>,
        claim: &crate::decoder::HybridCudaPreparationClaim,
    ) -> Result<Self> {
        Self::prepare_rank_inner(
            resources,
            plan,
            parallel,
            layout,
            options,
            device,
            collective,
            experts,
            Some(claim),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_rank_inner(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        parallel: HybridParallelPlan,
        layout: Arc<HybridRankLayout>,
        options: &GenericDecoderOptions,
        device: HybridCudaDevice,
        collective: Box<dyn StandardTensorCollective>,
        experts: Option<HybridCudaRoutedExperts>,
        shared_claim: Option<&crate::decoder::HybridCudaPreparationClaim>,
    ) -> Result<Self> {
        let mut own_claim = None;
        let mut external = ExternalExpertCustody::new(experts, &device);
        let mut collective = Some(collective);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let memory = CudaHybridRankMemory::for_rank(
                resources,
                &plan,
                &parallel,
                &layout,
                options,
                device.budget().kv_pages,
            )?;
            if collective.as_ref().unwrap().owner() != layout.owner()
                || collective.as_ref().unwrap().members() != parallel.tensor_members()
                || external.attachment.is_some()
                    != parallel
                        .expert_attachment()
                        .is_some_and(|ep| ep.source_rank == layout.owner())
                || external.attachment.as_ref().is_some_and(|a| {
                    Some(a.source_rank) != parallel.expert_attachment().map(|ep| ep.source_rank)
                })
            {
                return Err(cuda_error(
                    "CUDA hybrid rank collective/EP source ownership mismatch",
                ));
            }
            memory.validate_device_budget(device.budget(), options.capabilities().max_sequences)?;
            if shared_claim.is_none() {
                own_claim = Some(device.admit(memory.admission_estimate()?, memory.max_sequences)?);
            }
            let claim = shared_claim
                .or(own_claim.as_ref())
                .expect("admitted rank image");
            claim.validate_estimate(memory.admission_estimate()?, memory.max_sequences)?;
            claim.begin_image(&device)?;
            let precision = memory.precision;
            let scratch = memory.numeric_scratch_limit;
            let materializer = StateDictMaterializer::for_hybrid_cuda_rank(
                options.max_parameter_bytes(),
                Arc::clone(&layout),
                memory.weight_grants,
            )?;
            let segment = HybridDecoderSegment::prepare_with_materializer(
                resources,
                plan,
                ExecutionPrecisionPolicy::f32(),
                options.max_positions(),
                materializer,
                Some(layout.state_schema().clone()),
            )
            .map_err(segment_error)?;
            let parameters = segment
                .parameters()
                .iter()
                .filter(|p| !matches!(p.residency(), ParameterResidency::Expert { .. }))
                .cloned()
                .collect::<Vec<_>>();
            let mut operators = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
                Rc::clone(device.operators()),
                &parameters,
                ExpertCachePolicy::KeepAll,
                scratch,
                precision,
            )?;
            operators.configure_numeric_fp8_rank(
                Arc::clone(&layout),
                memory.weight_grants + memory.numeric_workspace_grants,
            )?;
            let channel = Rc::new(RefCell::new(RankChannel::new(
                parallel.collective_schedule(segment.plan())?,
                collective.take().expect("rank transport"),
            )));
            operators.hybrid_rank = Some(RankOperators {
                layout: Arc::clone(&layout),
                channel: Rc::clone(&channel),
                max_rows: options.capabilities().max_batch_tokens,
                linears: BTreeMap::new(),
            });
            let before = device.operators().allocator_metrics().granted_bytes;
            let prepared = segment.prepare_cuda_bindings(&mut operators);
            if operators.needs_quarantine() {
                device.quarantine();
            }
            prepared?;
            device.fence()?;
            let actual = device
                .operators()
                .allocator_metrics()
                .granted_bytes
                .checked_sub(before)
                .ok_or_else(|| cuda_error("rank allocator accounting overflow"))?;
            if actual > (memory.weight_grants + memory.numeric_workspace_grants) as u64 {
                return Err(cuda_error(
                    "rank actual physical weight/workspace grants exceed admission",
                ));
            }
            let expert_boundary = Boundary::new(device.operators())?;
            claim.image_prepared()?;
            Ok((segment, operators, channel, memory, expert_boundary))
        }));
        let result = match result {
            Ok(Ok((segment, operators, channel, memory, expert_boundary))) => Ok(Self {
                state_credits: lifecycle::StateCredits::new(
                    (2 * memory.max_sequences + 1) * memory.per_sequence_state_grants,
                    memory.per_sequence_state_grants,
                ),
                segment,
                parallel,
                layout,
                channel,
                operators,
                device,
                external,
                memory,
                expert_boundary,
                watermark: None,
                failed: false,
                closed: false,
                shutdown_started: false,
            }),
            Ok(Err(primary)) => {
                if let Some(transport) = &mut collective {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        transport.abort()
                    }));
                }
                let cleanup = external.require_drain(&device, true);
                if cleanup.is_ok() {
                    external.close();
                } else {
                    device.quarantine();
                    external.retain_unknown();
                    std::mem::forget(device.clone());
                }
                drop(external);
                Err(Error::with_cleanup(
                    "CUDA rank preparation",
                    primary,
                    cleanup,
                ))
            }
            Err(panic) => {
                if let Some(transport) = &mut collective {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        transport.abort()
                    }));
                }
                device.quarantine();
                external.retain_unknown();
                std::mem::forget(device.clone());
                std::panic::resume_unwind(panic)
            }
        };
        match own_claim {
            Some(claim) => claim.finish(result),
            None => result,
        }
    }
    pub fn is_failed(&self) -> bool {
        self.failed
    }
    pub fn retains_expert_custody(&self) -> bool {
        self.external.forward.is_some()
            || self.external.kv.is_some()
            || !self.external.holds.is_empty()
    }
    pub fn collective_schedule(&self) -> HybridCollectiveSchedule {
        self.channel.borrow().schedule.clone()
    }
    pub fn layout(&self) -> &HybridRankLayout {
        &self.layout
    }
    pub fn plan(&self) -> &LayerSegmentPlan {
        self.segment.plan()
    }
    pub fn parallel_plan(&self) -> &HybridParallelPlan {
        &self.parallel
    }
    pub fn schema(&self) -> &HybridStateSchema {
        self.layout.state_schema()
    }
    pub fn memory(&self) -> &CudaHybridRankMemory {
        &self.memory
    }
    pub fn operators(&self) -> &CudaStandardDecoderOperators {
        &self.operators
    }
    pub fn new_lifecycle(&self) -> Result<CudaHybridRankLifecycle> {
        if self.closed || self.shutdown_started {
            return Err(cuda_error("closed CUDA rank"));
        }
        CudaHybridRankLifecycle::new(
            self.schema().clone(),
            self.device.clone(),
            Rc::clone(&self.state_credits),
        )
    }
    pub fn needs_quarantine(&self) -> bool {
        self.device.needs_quarantine()
            || self.operators.needs_quarantine()
            || self.expert_boundary.needs_quarantine()
    }
    pub fn validate_frontier(
        &self,
        batch: &PackedDecoderBatch,
        states: &[CudaHybridRankState],
    ) -> Result<()> {
        for state in states {
            state.validate_rank(&self.state_credits)?;
        }
        self.segment
            .validate_frontier(batch, states)
            .map_err(segment_error)
    }
    pub fn poll_quiescence(&mut self) -> Result<bool> {
        self.external.begin_drain()?;
        if !self.external.drain(&self.device, false)? {
            return Ok(false);
        }
        let proof = self.operators.quiesce();
        if self.operators.needs_quarantine() {
            self.device.quarantine();
        }
        proof?;
        self.expert_boundary.drain()?;
        if self.expert_boundary.needs_quarantine() {
            self.device.quarantine();
        }
        self.device.fence()?;
        self.external.finish();
        if self.layout.members().len() == 1 {
            self.failed = false;
        }
        Ok(true)
    }
    pub fn poll_shutdown(&mut self) -> Result<bool> {
        if !self.shutdown_started {
            self.channel.borrow_mut().abort();
        }
        self.shutdown_started = true;
        if self.closed {
            self.device.ensure_ready()?;
            return Ok(true);
        }
        self.external.begin_drain()?;
        if !self.external.drain(&self.device, true)? {
            return Ok(false);
        }
        let proof = self.operators.quiesce();
        if self.operators.needs_quarantine() {
            self.device.quarantine();
        }
        proof?;
        self.expert_boundary.drain()?;
        if self.expert_boundary.needs_quarantine() {
            self.device.quarantine();
        }
        self.device.fence()?;
        self.external.close();
        self.closed = true;
        Ok(true)
    }
    pub fn shutdown(&mut self) -> Result<()> {
        if self.poll_shutdown()? {
            Ok(())
        } else {
            Err(cuda_error("rank shutdown pending; retain custody"))
        }
    }
}
mod execution;
impl Drop for CudaHybridRankSegment {
    fn drop(&mut self) {
        let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.poll_shutdown()));
        if !matches!(done, Ok(Ok(true))) || self.needs_quarantine() {
            self.device.quarantine();
            self.operators.poisoned = true;
            self.external.retain_unknown();
            std::mem::forget(self.device.clone());
        }
    }
}
