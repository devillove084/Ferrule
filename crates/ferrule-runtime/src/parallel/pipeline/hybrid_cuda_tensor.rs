//! CUDA hybrid rank binding only. The existing PP journal owns state and KV custody.
use super::completion_boundary::checked_cuda_completion;
use super::tensor_scope::TensorScopedStateProgram;
use super::*;
use crate::parallel::collective::HostCollectiveLimits;
use crate::parallel::hybrid_topology::HybridExpertTopology;
use crate::parallel::tensor::decoder_collective::DecoderTensorCollective;
use ferrule_common::{Error, ParallelGroupId};
use ferrule_model::decoder::{
    CudaKvView, HybridCudaDevice, HybridCudaRoutedExecutor, HybridCudaRoutedExperts,
    HybridLayerSchema, HybridStateSchema, PackedDecoderBatch, TypedCudaPagedKvPool, WorkProgress,
};
use ferrule_model::transformer::hybrid_parallel::HybridParallelPlan;
use ferrule_model::transformer::{
    CudaHybridRankLifecycle, CudaHybridRankMemory, CudaHybridRankSegment, CudaHybridRankState,
    DecoderModelSpec, SegmentOutput,
};
use std::sync::Mutex;

#[path = "mixed_cuda_tensor.rs"]
mod mixed;
pub use mixed::MixedCudaTensorProgram;

mod leaf_seal {
    pub trait Sealed {}
    impl Sealed for super::HybridCudaTensorProgram {}
    impl Sealed for super::MixedCudaTensorProgram {}
}
/// Only concrete owner leaves may select the typed execution protocol. Arbitrary
/// state-program wrappers cannot use BF16 metadata to bypass numeric admission.
pub trait HybridCudaTensorLeaf:
    leaf_seal::Sealed + PipelineStateProgram<SequenceState = CudaHybridRankState, KvView = CudaKvView>
{
    const MIXED: bool;
}
impl HybridCudaTensorLeaf for HybridCudaTensorProgram {
    const MIXED: bool = false;
}

pub struct HybridCudaTensorProgram {
    segment: CudaHybridRankSegment,
    device: HybridCudaDevice,
    group: Option<ExpertGroup>,
    lifecycle: Option<CudaHybridRankLifecycle>,
}

fn checked<T>(device: &HybridCudaDevice, result: Result<T>) -> Result<T> {
    checked_cuda_completion(device.needs_quarantine(), result)
}
// Before a worker exists there is no panic hook to retain its model custody.
// Model/allocator unwind guards retain unsafe buffers; preserve that Unknown
// evidence at the runtime owner boundary instead of reporting plain Panicked.
fn cuda_call<T>(call: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)) {
        Ok(value) => value,
        Err(payload) => {
            std::mem::forget(payload);
            std::panic::panic_any(PanicQuiescence::Unknown);
        }
    }
}
type ReadyRankExperts = Option<(ExpertGroup, Box<dyn HybridCudaRoutedExecutor>)>;

// Before model attachment, no device claim can retain or report EP custody.
// Only explicit shutdown proof permits dropping the executor, including unwind.
struct StartupExperts(ReadyRankExperts);
impl Drop for StartupExperts {
    fn drop(&mut self) {
        std::mem::forget(self.0.take());
    }
}
impl StartupExperts {
    fn shutdown(&mut self, timeout: Duration) -> Result<()> {
        let Some((_, executor)) = self.0.as_mut() else {
            return Ok(());
        };
        let deadline = std::time::Instant::now() + timeout;
        let mut drained = false;
        loop {
            let progress = if drained {
                executor.shutdown()?
            } else {
                executor.drain()?
            };
            match progress {
                WorkProgress::Complete if drained => break,
                WorkProgress::Complete => drained = true,
                WorkProgress::Unknown => {
                    return Err(error("rank startup expert cleanup completion unknown"));
                }
                WorkProgress::Pending => {
                    if std::time::Instant::now() >= deadline {
                        return Err(error("rank startup expert cleanup timed out"));
                    }
                    std::thread::sleep(Duration::from_micros(50));
                }
            }
        }
        drop(self.0.take());
        Ok(())
    }
}

fn prepare_rank_startup<T>(
    routed: ReadyRankExperts,
    timeout: Duration,
    prepare: impl FnOnce(Option<&ExpertGroup>) -> Result<T>,
) -> Result<(T, ReadyRankExperts)> {
    let mut custody = StartupExperts(routed);
    cuda_call(|| {
        match prepare(custody.0.as_ref().map(|(group, _)| group)) {
            Ok(value) => Ok((value, custody.0.take())),
            Err(primary) => {
                // An error/panic/timeout during cleanup is not a retirement ACK.
                // checked_cuda_completion forces the parent initialization path
                // to preserve Unknown; the guard retains the unbound executor.
                if let Err(cleanup) = custody.shutdown(timeout) {
                    return checked_cuda_completion(
                        true,
                        Err(Error::with_cleanup(
                            "rank startup experts",
                            primary,
                            Err(cleanup),
                        )),
                    );
                }
                Err(primary)
            }
        }
    })
}

impl HybridCudaTensorProgram {
    fn completion<T>(&self, result: Result<T>) -> Result<T> {
        checked_cuda_completion(
            self.segment.needs_quarantine() || self.device.needs_quarantine(),
            result,
        )
    }
}
impl PipelineStateProgram for HybridCudaTensorProgram {
    type SequenceState = CudaHybridRankState;
    type Lifecycle = CudaHybridRankLifecycle;
    type KvView = CudaKvView;

    fn plan(&self) -> &LayerSegmentPlan {
        self.segment.plan()
    }
    fn hybrid_schema(&self) -> Option<&HybridStateSchema> {
        Some(self.segment.schema())
    }
    fn hybrid_expert_group(&self) -> Option<&ExpertGroup> {
        self.group.as_ref()
    }
    fn poll_quiescence(&mut self) -> Result<bool> {
        let result = cuda_call(|| self.segment.poll_quiescence());
        self.completion(result)
    }
    fn new_lifecycle(&mut self) -> Result<Self::Lifecycle> {
        if let Some(lifecycle) = self.lifecycle.take() {
            return Ok(lifecycle);
        }
        self.completion(self.segment.new_lifecycle())
    }
    fn execute_state(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [Self::SequenceState],
        view: &mut CudaKvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        if context.experts.is_some()
            || context.transaction != view.transaction()
            || context.sequence_ids.len() != batch.sequences().len()
        {
            return Err(error(
                "hybrid CUDA execution identity mismatch or unbound expert injection",
            ));
        }
        (context.check_active)(context.transaction)?;
        let result = cuda_call(|| {
            self.segment.execute(
                batch,
                input,
                states,
                view,
                context.sequence_ids,
                context.check_active,
            )
        });
        self.completion(result)
    }

    fn validate_frontier(
        &self,
        batch: &PackedDecoderBatch,
        states: &[Self::SequenceState],
    ) -> Result<()> {
        self.completion(self.segment.validate_frontier(batch, states))
    }
    fn shutdown(&mut self) -> Result<()> {
        // The thread worker has a synchronous shutdown boundary, not a second
        // completion journal. Poll model custody to proof within the existing
        // thread expert deadline; timeout is Unknown, never an ordinary ACK.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let result = cuda_call(|| self.segment.poll_shutdown());
            if self.completion(result)? {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                return checked_cuda_completion(
                    true,
                    Err(error("hybrid root/expert shutdown completion unknown")),
                );
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }
}
pub type HybridCudaTensorStage<P = HybridCudaTensorProgram> = PipelineStage<
    PagedKvBackend<TypedCudaPagedKvPool<CudaHybridRankState>>,
    TensorScopedStateProgram<P>,
>;

fn description(
    spec: &DecoderModelSpec,
    plan: LayerSegmentPlan,
    schema: &HybridStateSchema,
    config: PipelineConfig,
    group: Option<ExpertGroup>,
) -> Result<PipelineStageDescription> {
    let (kv_heads, head_dim) = schema
        .layers()
        .iter()
        .find_map(|l| match l {
            HybridLayerSchema::FullAttention {
                kv_heads, head_dim, ..
            } => Some((*kv_heads, *head_dim)),
            _ => None,
        })
        .ok_or_else(|| error("hybrid CUDA rank has no full-attention KV plane"))?;
    Ok(PipelineStageDescription {
        plan,
        config,
        physical_pages: config
            .physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda))?,
        hidden: spec.hidden_size(),
        vocabulary: spec.vocab_size(),
        kv_heads,
        head_dim,
        expert_group: group,
    })
}
impl HybridCudaTensorStage {
    /// Takes ready runtime executors before wrapping them in model attachments,
    /// so even pre-claim failures must cross an explicit expert cleanup barrier.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_hybrid_cuda_tensor(
        resources: &BoundDecoderResources,
        segment: LayerSegmentPlan,
        parallel: HybridParallelPlan,
        lane: usize,
        config: PipelineConfig,
        options: &GenericDecoderOptions,
        device: HybridCudaDevice,
        endpoint: DecoderTensorCollective,
        routed: Option<(ExpertGroup, Box<dyn HybridCudaRoutedExecutor>)>,
    ) -> Result<Self> {
        let ((layout, owner, description, capabilities, planes, control, claim), routed) =
            prepare_rank_startup(routed, Duration::from_secs(30), |group| {
                config.validate()?;
                if options.page_size() != config.page_size
                    || options.max_positions() != config.max_positions
                    || options.max_parameter_bytes() != config.max_parameter_bytes
                    || options.capabilities().max_batch_tokens != config.max_batch_tokens
                    || options.capabilities().max_sequences != config.session_capacity
                    || options.precision() != config.precision
                    || config.precision != ExecutionPrecisionPolicy::f32()
                {
                    return Err(error("hybrid CUDA rank options differ from admission"));
                }
                let layout = Arc::new(parallel.rank_layout(lane)?);
                let owner = layout.owner();
                if let Some(group) = group {
                    group.validate_resources(resources)?;
                    let attachment = parallel
                        .expert_attachment()
                        .ok_or_else(|| error("unexpected rank experts"))?;
                    if group.source_rank != owner
                        || group.source_rank != attachment.source_rank
                        || group.members != attachment.members
                        || group.placement != attachment.placement
                        || group.layers != segment.layers()
                    {
                        return Err(error("rank experts differ from sealed stage attachment"));
                    }
                }
                let description = description(
                    resources.spec(),
                    segment.clone(),
                    layout.state_schema(),
                    config,
                    group.cloned(),
                )?;
                let capabilities = description.execution_capabilities()?;
                let planes = layout.kv_planes(config.page_size, config.max_positions)?;
                let memory = CudaHybridRankMemory::for_rank(
                    resources,
                    &segment,
                    &parallel,
                    &layout,
                    options,
                    config.physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda))?,
                )?;
                let control = endpoint.control();
                control.register_sites(owner, translate_hybrid_schedule(&parallel, &segment)?)?;
                // One opaque claim covers image, pool and lifecycle. Never expose this device
                // to another factory, even when several owner contexts share a physical GPU.
                let claim = device.admit(memory.admission_estimate()?, config.session_capacity)?;
                Ok((
                    layout,
                    owner,
                    description,
                    capabilities,
                    planes,
                    control,
                    claim,
                ))
            })?;
        let result = cuda_call(|| {
            (|| {
                let module = CudaHybridRankSegment::prepare_rank_with_claim(
                    resources,
                    segment,
                    parallel,
                    layout,
                    options,
                    device.clone(),
                    Box::new(endpoint),
                    routed.map(|(group, executor)| {
                        HybridCudaRoutedExperts::new(group.source_rank, executor)
                    }),
                    &claim,
                )?;
                let lifecycle = Some(module.new_lifecycle()?);
                let mut program = HybridCudaTensorProgram {
                    segment: module,
                    device: device.clone(),
                    lifecycle,
                    group: description.expert_group.clone(),
                };
                let pool = match TypedCudaPagedKvPool::from_strategy(
                    Rc::clone(device.operators()),
                    &planes,
                    config.physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda))?,
                ) {
                    Ok(pool) => pool,
                    Err(primary) => {
                        return Err(Error::with_cleanup(
                            "rank KV preparation",
                            primary,
                            program.shutdown(),
                        ));
                    }
                };
                Self::new(
                    TensorScopedStateProgram::new(program, control, owner),
                    PagedKvBackend::new(pool),
                    description,
                    capabilities,
                )
            })()
        });
        checked(&device, claim.finish(result))
    }
}

impl PipelineParallelExecutor {
    /// Authenticate TP and optional external EP scopes, then use the ordinary worker pool,
    /// transaction journal and sole logical page authority. The state-program leaf is
    /// generic; only the sealed experimental Mixed leaf selects BF16. The legacy
    /// F32 leaf and all other pipeline constructors retain their existing gates.
    #[allow(clippy::too_many_arguments)]
    pub fn new_hybrid_cuda_tensor<F, P>(
        topology: ValidatedParallelTopology,
        resources: Arc<BoundDecoderResources>,
        plans: Vec<LayerSegmentPlan>,
        layout: HybridExpertTopology,
        groups: Vec<ExpertGroup>,
        config: PipelineConfig,
        limits: HostCollectiveLimits,
        timeout: Duration,
        build: F,
    ) -> Result<Self>
    where
        F: Fn(
                PipelineRank,
                LayerSegmentPlan,
                HybridParallelPlan,
                usize,
                DecoderTensorCollective,
            ) -> Result<HybridCudaTensorStage<P>>
            + Send
            + Sync
            + 'static,
        P: HybridCudaTensorLeaf,
    {
        if P::MIXED {
            MixedTensorExecutionPlan::new(
                resources.spec(),
                &topology,
                &plans,
                &layout,
                config,
                limits,
            )?;
            if !groups.is_empty() {
                return Err(error("Mixed local experts cannot attach external owners"));
            }
        } else {
            validate_pipeline_mesh(&topology, &plans, config, true)?;
        }
        let tp = topology.plan().tensor_parallel;
        if !matches!(tp, 2 | 4) || layout.stages().len() != plans.len() {
            return Err(error("CUDA hybrid TP layout/config mismatch"));
        }
        let scopes = layout.bind_tensor_scopes(&topology, &plans, &groups)?;
        // All metadata/provider capability checks precede the first persistent worker.
        // A generic state program does not authorize a new numeric profile.
        for parallel in layout.stages() {
            parallel.validate_spec(resources.spec())?;
            if !P::MIXED {
                parallel.validate_rank_execution()?;
            }
        }
        for group in &groups {
            group.validate_resources(&resources)?;
        }
        let mut endpoints = Vec::new();
        let mut controls = Vec::new();
        let mut expected = Vec::new();
        let mut kv_layers = Vec::new();
        for (stage, parallel) in layout.stages().iter().enumerate() {
            let group = groups.iter().find(|g| g.layers == plans[stage].layers());
            let peers = DecoderTensorCollective::new_group_from_members(
                parallel.tensor_members(),
                topology.topology_id(),
                ParallelGroupId::new(stage as u32 + 1),
                limits,
                timeout,
            )?;
            controls.push(peers[0].control());
            endpoints.extend(peers.into_iter().map(Some));
            for lane in 0..tp {
                let local = parallel.rank_layout(lane)?;
                expected.push(description(
                    resources.spec(),
                    plans[stage].clone(),
                    local.state_schema(),
                    config,
                    group.filter(|g| g.source_rank == local.owner()).cloned(),
                )?);
                kv_layers.push(
                    local
                        .state_schema()
                        .entries()
                        .iter()
                        .filter(|e| e.local_kv_layer.is_some())
                        .count(),
                );
            }
        }
        let logical_schema =
            HybridStateSchema::from_spec(resources.spec(), resources.spec().layers().len())?
                .kv_planes_with_dtype(config.page_size, config.max_positions, config.dtype())?;
        let endpoints = Arc::new(Mutex::new(endpoints));
        let admitted = expected.clone();
        let build = Arc::new(build);
        let degree = topology.world_size() as usize;
        let pool = DataParallelExecutor::new(
            DataParallelConfig {
                replicas: degree,
                max_outstanding_per_replica: 1,
                session_capacity: degree,
            },
            move |local| {
                let index = local.get() as usize;
                let rank = Self::rank(index);
                if !resources.state_dict().validate_source_identities() {
                    return Err(error("hybrid CUDA immutable checkpoint seal is stale"));
                }
                let endpoint = endpoints
                    .lock()
                    .map_err(|_| error("hybrid endpoint lock poisoned"))?[index]
                    .take()
                    .ok_or_else(|| error("hybrid endpoint assigned twice"))?;
                let segment = plans[index / tp].clone();
                let parallel = layout.stages()[index / tp].clone();
                let mut owner = build(
                    rank,
                    segment.clone(),
                    parallel.clone(),
                    index % tp,
                    endpoint,
                )?;
                if owner.description != admitted[index]
                    || owner.program.hybrid_schema()
                        != Some(parallel.rank_layout(index % tp)?.state_schema())
                {
                    return Err(Error::with_cleanup(
                        "CUDA rank identity",
                        error("returned owner differs from sealed layout"),
                        owner.shutdown(),
                    ));
                }
                let boot = PipelineStageBoot {
                    rank,
                    plan: segment,
                    config,
                    physical_pages: config
                        .physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda))?,
                    program_spec: vec![],
                };
                Ok(PipelineStageWorker::new(&boot, owner)?.boxed())
            },
        )
        .map_err(pool_build_error)?;
        let mut transport = ThreadTransport::new(pool);
        let validation = (|| {
            for (index, expected) in expected.iter().enumerate() {
                match transport.call(Self::rank(index), Command::Describe)? {
                    Reply::Description(actual) if actual == *expected => (),
                    _ => return Err(error("CUDA rank description differs from admission")),
                }
            }
            Ok(())
        })();
        if let Err(source) = validation {
            return Err(Error::with_cleanup(
                "CUDA hybrid parent startup",
                source,
                transport.shutdown(),
            ));
        }
        let mut executor = Self::from_validated_transport(
            topology,
            config,
            transport,
            expected,
            scopes,
            Box::new(logical_schema),
            kv_layers,
        );
        executor.tensor_controls = controls;
        Ok(executor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::data::{BuildError, OwnerFailureKind};
    use ferrule_model::transformer::expert_parallel::{
        ExpertDispatchLimits, ExpertPlacement, RoutedSwiGluExecutor, RoutedSwiGluRequest,
    };
    use ferrule_model::transformer::{OperatorProgress, Rows};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Copy)]
    enum Cleanup {
        Complete,
        Unknown,
        Pending,
        Error,
        Panic,
    }
    #[derive(Default)]
    struct Probe {
        drains: AtomicUsize,
        shutdowns: AtomicUsize,
        drops: AtomicUsize,
        sibling_shutdowns: AtomicUsize,
    }
    struct MockExperts {
        probe: Arc<Probe>,
        drain: Cleanup,
        shutdown: Cleanup,
    }
    fn progress(cleanup: Cleanup) -> Result<WorkProgress> {
        match cleanup {
            Cleanup::Complete => Ok(WorkProgress::Complete),
            Cleanup::Unknown => Ok(WorkProgress::Unknown),
            Cleanup::Pending => Ok(WorkProgress::Pending),
            Cleanup::Error => Err(error("injected cleanup error")),
            Cleanup::Panic => panic!("injected cleanup panic"),
        }
    }
    impl Drop for MockExperts {
        fn drop(&mut self) {
            self.probe.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl RoutedSwiGluExecutor for MockExperts {
        fn routed_swiglu(
            &mut self,
            _: RoutedSwiGluRequest<'_>,
            _: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
        ) -> Result<OperatorProgress<Rows>> {
            panic!("startup cleanup must not dispatch model work")
        }
    }
    impl HybridCudaRoutedExecutor for MockExperts {
        fn drain(&mut self) -> Result<WorkProgress> {
            self.probe.drains.fetch_add(1, Ordering::SeqCst);
            progress(self.drain)
        }
        fn shutdown(&mut self) -> Result<WorkProgress> {
            self.probe.shutdowns.fetch_add(1, Ordering::SeqCst);
            progress(self.shutdown)
        }
    }
    fn ready_experts(probe: Arc<Probe>, drain: Cleanup, shutdown: Cleanup) -> ReadyRankExperts {
        let source = ParallelRankId::new(7);
        let member = ParallelRankId::new(19);
        Some((
            ExpertGroup {
                source_rank: source,
                members: vec![member],
                layers: 0..1,
                placement: ExpertPlacement::new([(0, 0, member)]).unwrap(),
                limits: ExpertDispatchLimits {
                    max_tokens: 1,
                    max_bytes: 4,
                },
            },
            Box::new(MockExperts {
                probe,
                drain,
                shutdown,
            }),
        ))
    }
    struct Sibling(Arc<Probe>);
    impl ReplicaWorker<()> for Sibling {
        type Output = ();
        type Error = Error;
        fn execute(&mut self, _: WorkRequest<()>) -> Result<()> {
            panic!("startup test must not execute")
        }
        fn shutdown(&mut self) -> Result<()> {
            self.0.sibling_shutdowns.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }
    fn failed_startup(drain: Cleanup, shutdown: Cleanup) -> (BuildError<Error>, Arc<Probe>) {
        let probe = Arc::new(Probe::default());
        let observed = Arc::clone(&probe);
        let pool = DataParallelExecutor::new(
            DataParallelConfig {
                replicas: 2,
                max_outstanding_per_replica: 1,
                session_capacity: 2,
            },
            move |rank| {
                if rank.get() == 0 {
                    return Ok(Sibling(Arc::clone(&observed)));
                }
                // No HybridCudaDevice/claim exists. Exercise the same barrier
                // as config, attachment, memory, schedule and admit failures.
                let result = prepare_rank_startup::<()>(
                    ready_experts(Arc::clone(&observed), drain, shutdown),
                    Duration::ZERO,
                    |_| {
                        Err(Error::Execution {
                            message: "original startup failure".into(),
                        })
                    },
                );
                assert!(result.is_err());
                result.map(|_| Sibling(Arc::clone(&observed)))
            },
        );
        let failure = match pool {
            Err(failure) => failure,
            Ok(_) => panic!("failed startup published Ready"),
        };
        assert_eq!(probe.sibling_shutdowns.load(Ordering::SeqCst), 1);
        (failure, probe)
    }
    #[test]
    fn rank_startup_cleanup_complete_preserves_original_error_and_releases_custody() {
        let (failure, probe) = failed_startup(Cleanup::Complete, Cleanup::Complete);
        let BuildError::Owners(owners) = &failure else {
            panic!("missing owner failure")
        };
        assert_eq!(owners.len(), 1);
        assert!(matches!(&owners[0].kind,
            OwnerFailureKind::Initialization(Error::Execution { message })
                if message == "original startup failure"));
        let parent = pool_build_error(failure);
        assert!(!super::super::completion_boundary::is_unknown(&parent));
        assert_eq!(probe.drains.load(Ordering::SeqCst), 1);
        assert_eq!(probe.shutdowns.load(Ordering::SeqCst), 1);
        assert_eq!(probe.drops.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn rank_startup_cleanup_unknown_quarantines_parent_and_retains_unbound_custody() {
        for shutdown in [
            Cleanup::Unknown,
            Cleanup::Pending,
            Cleanup::Error,
            Cleanup::Panic,
        ] {
            let (failure, probe) = failed_startup(Cleanup::Complete, shutdown);
            let BuildError::Owners(owners) = &failure else {
                panic!("missing owner failure")
            };
            assert_eq!(owners.len(), 1);
            assert!(matches!(
                owners[0].kind,
                OwnerFailureKind::QuiescenceUnknown
            ));
            let parent = pool_build_error(failure);
            assert!(super::super::completion_boundary::is_unknown(&parent));
            assert!(super::super::completion_boundary::needs_quarantine(&parent));
            assert_eq!(probe.drains.load(Ordering::SeqCst), 1);
            assert_eq!(probe.shutdowns.load(Ordering::SeqCst), 1);
            assert_eq!(probe.drops.load(Ordering::SeqCst), 0);
        }
    }
    #[test]
    fn rank_startup_unproven_drain_never_calls_shutdown_or_releases_custody() {
        for drain in [
            Cleanup::Unknown,
            Cleanup::Pending,
            Cleanup::Error,
            Cleanup::Panic,
        ] {
            let (failure, probe) = failed_startup(drain, Cleanup::Complete);
            assert!(super::super::completion_boundary::is_unknown(
                &pool_build_error(failure)
            ));
            assert_eq!(probe.drains.load(Ordering::SeqCst), 1);
            assert_eq!(probe.shutdowns.load(Ordering::SeqCst), 0);
            assert_eq!(probe.drops.load(Ordering::SeqCst), 0);
        }
    }
    #[test]
    fn rank_startup_success_transfers_experts_without_cleanup() {
        let probe = Arc::new(Probe::default());
        let (value, routed) = prepare_rank_startup(
            ready_experts(Arc::clone(&probe), Cleanup::Complete, Cleanup::Complete),
            Duration::ZERO,
            |group| {
                assert_eq!(group.unwrap().source_rank, ParallelRankId::new(7));
                Ok(42)
            },
        )
        .unwrap();
        assert_eq!(value, 42);
        assert_eq!(probe.drains.load(Ordering::SeqCst), 0);
        assert_eq!(probe.shutdowns.load(Ordering::SeqCst), 0);
        assert_eq!(probe.drops.load(Ordering::SeqCst), 0);
        StartupExperts(routed).shutdown(Duration::ZERO).unwrap();
        assert_eq!(probe.drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn rank_startup_panic_preserves_unknown_without_a_device() {
        let probe = Arc::new(Probe::default());
        let payload = std::panic::catch_unwind(|| {
            prepare_rank_startup::<()>(
                ready_experts(Arc::clone(&probe), Cleanup::Complete, Cleanup::Complete),
                Duration::ZERO,
                |_| panic!("lost rank completion"),
            )
        })
        .err()
        .expect("startup panic must propagate");
        assert_eq!(
            payload.downcast_ref::<PanicQuiescence>(),
            Some(&PanicQuiescence::Unknown)
        );
        assert_eq!(probe.drains.load(Ordering::SeqCst), 0);
        assert_eq!(probe.shutdowns.load(Ordering::SeqCst), 0);
        assert_eq!(probe.drops.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn mixed_local_startup_failure_preserves_unknown_and_quiesces_ready_sibling() {
        use ferrule_model::decoder::CompletionUnknown;
        for unknown in [false, true] {
            let probe = Arc::new(Probe::default());
            let observed = Arc::clone(&probe);
            let pool = DataParallelExecutor::new(
                DataParallelConfig {
                    replicas: 2,
                    max_outstanding_per_replica: 1,
                    session_capacity: 2,
                },
                move |rank| {
                    if rank.get() == 0 {
                        return Ok(Sibling(Arc::clone(&observed)));
                    }
                    let primary = error("Mixed local image rejected");
                    let cleanup = if unknown {
                        Err(CompletionUnknown::wrap(error(
                            "Mixed claim cleanup unproven",
                        )))
                    } else {
                        Ok(())
                    };
                    checked_cuda_completion(
                        false,
                        Err(Error::with_cleanup("Mixed local startup", primary, cleanup)),
                    )
                },
            );
            let failure = match pool {
                Err(e) => pool_build_error(e),
                Ok(_) => panic!("failed owner published Ready"),
            };
            assert_eq!(probe.sibling_shutdowns.load(Ordering::SeqCst), 1);
            assert_eq!(
                super::super::completion_boundary::is_unknown(&failure),
                unknown
            );
            if !unknown {
                assert!(failure.to_string().contains("Mixed local image rejected"));
            }
        }
    }
}
