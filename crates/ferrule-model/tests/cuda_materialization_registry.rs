//! Real provider -> shared bridge -> resolver -> existing registry, not a mock.
//! Entry point (the model manifest owns the opt-in feature):
//! CARGO_INCREMENTAL=0 CUDA_VISIBLE_DEVICES=1 FERRULE_CUDA_ARCH=sm_86
//! cargo test -p ferrule-model --features cuda-test-support --test
//! cuda_materialization_registry -- --ignored --test-threads=1 --nocapture
//! Missing io_uring/CUDA/SM103 is a failure, never an early-return skip.
#[cfg(not(target_os = "linux"))]
#[test]
fn requires_linux_io_uring() {
    panic!("This opt-in real-provider composition target requires Linux io_uring; no fallback");
}

#[cfg(target_os = "linux")]
mod real {

    use std::path::PathBuf;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    };
    use std::time::{Duration, Instant};

    use ferrule_backend::cuda::operators::moe::{CudaArtifactLinearShape, CudaOperators};
    use ferrule_common::materialization_io::{
        MaterializationResourceLimits, MaterializationResourceRequirements,
    };
    use ferrule_common::*;
    use ferrule_model::cuda_test_support::CudaExpertMaterializationProvider;
    use ferrule_model::moe::streaming::ExpertSourceCatalog;
    use ferrule_model::{
        ExpertIoTransport, ExpertLoadSource, ExpertMatrixKind, ExpertStreamingReader,
        ExpertTensorComponent, ExpertTensorKey, ExpertTensorSlice, MaterializationPlacement,
        MaterializationRequest, MaterializationResolver, MaterializationSourceCatalog,
        ResourceRetention,
    };
    use ferrule_runtime::expert_residency::ExpertResidencyController;
    use ferrule_runtime::io::{
        FairQueueConfig, LoadRegistry, LoadRequest, RuntimeMaterializationProvider,
        RuntimeMaterializationResolver, SharedMaterializationProvider,
    };
    use ferrule_runtime::scheduling::{
        ExecutionPhase, PhysicalResourceBroker, PhysicalResourceLimit, ResourceDemand, ResourceKind,
    };

    type Registry = LoadRegistry<SharedMaterializationProvider>;
    const MODEL: u64 = 722;
    const BYTES: u64 = 21_760;
    const BUFFER: usize = 16_384;
    const SLABS: usize = 8;

    fn gpu() -> CudaOperators {
        assert_eq!(
            std::env::var("CUDA_VISIBLE_DEVICES").as_deref(),
            Ok("1"),
            "strict GPU1 lane: set CUDA_VISIBLE_DEVICES=1 before launching this test"
        );
        let ops =
            CudaOperators::new_on_device(0).expect("physical GPU1 must be available; no skip");
        assert_eq!(
            ops.device_ordinal(),
            0,
            "visible ordinal 0 is physical GPU1"
        );
        ops
    }

    fn require_fp4(supported: bool) {
        use ferrule_backend::cuda::providers::{COMPILED_TARGET, compiled_capabilities};
        let caps = compiled_capabilities();
        eprintln!("compiled_target={COMPILED_TARGET:?} capabilities={caps:?}");
        assert_eq!(
            caps.sm103_block_scaled_fp4, supported,
            "UNSUPPORTED_ENVIRONMENT: success needs SM103 grouped-FP4; the rejection contract needs an unsupported build. Neither substitutes for the other"
        );
    }

    // Delegate the existing controller; inject only a one-shot cancellation error.
    // No slots, prepared tokens, publication or lease accounting are fabricated.
    struct ControlState {
        inner: ExpertResidencyController,
        fail_cancel: usize,
        attempts: Vec<PreparedExpertInstall>,
        activations: usize,
    }
    #[derive(Clone)]
    struct Control(Arc<Mutex<ControlState>>);
    impl Control {
        fn new() -> Self {
            Self(Arc::new(Mutex::new(ControlState {
                inner: ExpertResidencyController::new(MODEL, [1]).unwrap(),
                fail_cancel: 0,
                attempts: Vec::new(),
                activations: 0,
            })))
        }
    }
    impl ExpertResidencyControl for Control {
        fn requirements(&self) -> ExpertResidencyRequirements {
            self.0.lock().unwrap().inner.requirements()
        }
        fn binding(&self, key: ExpertKey) -> Result<Option<ExpertSlotBinding>> {
            self.0.lock().unwrap().inner.binding(key)
        }
        fn acquire_selected(&mut self, key: ExpertKey) -> Result<Option<ExpertResidencyGrant>> {
            self.0.lock().unwrap().inner.acquire_selected(key)
        }
        fn release(&mut self, lease: ExpertLease) -> Result<()> {
            self.0.lock().unwrap().inner.release(lease)
        }
        fn prepare_install(
            &mut self,
            intent: ExpertInstallIntent,
        ) -> Result<ExpertInstallPrepareOutcome> {
            self.0.lock().unwrap().inner.prepare_install(intent)
        }
        fn promote_install(
            &mut self,
            prepared: PreparedExpertInstall,
        ) -> Result<PreparedExpertInstall> {
            self.0.lock().unwrap().inner.promote_install(prepared)
        }
        fn activate_install(
            &mut self,
            prepared: PreparedExpertInstall,
        ) -> Result<ExpertInstallActivationOutcome> {
            let mut state = self.0.lock().unwrap();
            let result = state.inner.activate_install(prepared)?;
            if matches!(result, ExpertInstallActivationOutcome::Activated) {
                state.activations += 1;
            }
            Ok(result)
        }
        fn publish_install(
            &mut self,
            prepared: PreparedExpertInstall,
        ) -> Result<ExpertResidencyGrant> {
            self.0.lock().unwrap().inner.publish_install(prepared)
        }
        fn cancel_install(&mut self, prepared: PreparedExpertInstall) -> Result<()> {
            let mut state = self.0.lock().unwrap();
            state.attempts.push(prepared);
            if state.fail_cancel != 0 {
                state.fail_cancel -= 1;
                return Err(Error::Execution {
                    message: "test: exact prepared-token cancellation failed".into(),
                });
            }
            state.inner.cancel_install(prepared)
        }
        fn stats(&self) -> ExpertResidencyStats {
            self.0.lock().unwrap().inner.stats()
        }
    }

    struct Fixture {
        dir: PathBuf,
        sources: Arc<MaterializationSourceCatalog<ExpertLoadSource>>,
        request: MaterializationRequest,
    }
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "ferrule-real-registry-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&dir).unwrap();
            let path = dir.join("expert.bin");
            let mut data = Vec::new();
            let mut tensors = Vec::new();
            // Same MXFP4 shape/encoding as the existing cuda_payload fixture.
            // Alignment padding belongs to storage, not the H2D payload.
            for (matrix, out, input) in [
                (ExpertMatrixKind::Gate, 128, 128),
                (ExpertMatrixKind::Up, 128, 128),
                (ExpertMatrixKind::Down, 64, 128),
            ] {
                for (component, dtype, columns, value) in [
                    (ExpertTensorComponent::Weight, "I8", input / 2, 0x22),
                    (ExpertTensorComponent::Scale, "F8_E8M0", input / 32, 127),
                ] {
                    let offset = data.len();
                    let bytes = out * columns;
                    data.resize(offset + bytes, value);
                    tensors.push(ExpertTensorSlice {
                        key: ExpertTensorKey::new(0, 0, matrix),
                        component,
                        path: path.clone(),
                        offset: offset as u64,
                        bytes: bytes as u64,
                        dtype: dtype.into(),
                        shape: vec![out, columns],
                    });
                    data.resize(data.len().next_multiple_of(4096), 0);
                }
            }
            std::fs::write(&path, data).unwrap();
            let checkpoint_tensors = tensors
                .iter()
                .map(|tensor| ferrule_model::CheckpointTensorSlice {
                    name: format!("{:?}.{:?}", tensor.key.matrix, tensor.component),
                    role: match tensor.key.matrix {
                        ExpertMatrixKind::Gate => ferrule_model::TensorRole::RoutedExpertGate,
                        ExpertMatrixKind::Up => ferrule_model::TensorRole::RoutedExpertUp,
                        ExpertMatrixKind::Down => ferrule_model::TensorRole::RoutedExpertDown,
                    },
                    path: tensor.path.clone(),
                    offset: tensor.offset,
                    bytes: tensor.bytes,
                    dtype: ferrule_model::CheckpointDType::from_safetensors_dtype(&tensor.dtype),
                    shape: tensor.shape.clone(),
                })
                .collect::<Vec<_>>();
            let bundle = ferrule_model::CheckpointSourceCatalog::capture(&checkpoint_tensors)
                .unwrap()
                .bundle_source(
                    b"registry-composition-mxfp4",
                    PayloadEncodingId::new(1),
                    &checkpoint_tensors,
                )
                .unwrap();
            let source = bundle.source();
            let descriptor = ExpertLoadSource::HfLocalTensorSet {
                tensors,
                source_files: Arc::from(bundle.source_files()),
            };
            assert_eq!(descriptor.bytes(), BYTES);
            let sources = ExpertSourceCatalog::from_resource_sources([(
                ferrule_model::ExpertId::new(0, 0),
                descriptor,
                source,
            )])
            .materialization_sources()
            .unwrap();
            let request = MaterializationRequest::for_placement(
                placement(),
                source,
                MaterializedResourceId::routed_expert(
                    LayerId::new(0),
                    ferrule_common::ExpertId::new(0),
                ),
            )
            .unwrap();
            Self {
                dir,
                sources: Arc::new(sources),
                request,
            }
        }
        fn provider(
            &self,
            ops: &CudaOperators,
            pinned: bool,
            cleanup_fault: bool,
            control: Control,
        ) -> SharedMaterializationProvider {
            let reader = if pinned {
                ExpertStreamingReader::with_cuda_pinned_for_test(BUFFER as u64, 8, BUFFER, SLABS,
                ExpertIoTransport::DirectIoUring, ops.pinned_host_allocator(), CompletionHub::new())
                .expect("UNSUPPORTED_ENVIRONMENT: real CUDA-pinned io_uring required (ENOSYS is a failure, no fallback)")
            } else {
                ExpertStreamingReader::new(BUFFER as u64)
            };
            let provider = CudaExpertMaterializationProvider::new_for_test(
                placement(),
                MaterializationResourceLimits {
                    capacity: MaterializationResourceRequirements {
                        read_slots: 8,
                        storage_read_bytes: (BUFFER * SLABS) as u64,
                        pinned_host_bytes: (BUFFER * SLABS) as u64,
                        upload_slots: 1,
                        h2d_bytes: BYTES,
                        install_slots: 1,
                        device_install_bytes: BYTES,
                    },
                    execution_reserve: Default::default(),
                },
                self.sources.clone(),
                reader,
                1,
                &[(0, 1)],
                ops.compute_stream_authority(),
                Box::new(control),
            )
            .expect("construct the real CUDA materialization provider");
            if cleanup_fault {
                provider.failpoints_for_test().arm_stream_sync();
            }
            SharedMaterializationProvider::new(Box::new(provider))
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
    fn placement() -> MaterializationPlacement {
        MaterializationPlacement::new(
            ModelInstanceId::new(MODEL),
            BackendId::new(1),
            DeviceId::new(0),
        )
        .unwrap()
    }
    fn registry(provider: SharedMaterializationProvider, waiters: u64) -> Registry {
        let topology = provider.resource_topology().unwrap();
        let limits = topology.stage_limits();
        let c = limits.capacity;
        let broker = PhysicalResourceBroker::new(ResourceKind::ALL.map(|kind| {
            let capacity = match kind {
                ResourceKind::ReadSlot => c.read_slots,
                ResourceKind::StorageReadBytes => c.storage_read_bytes,
                ResourceKind::PinnedHostBytes => c.pinned_host_bytes,
                ResourceKind::UploadSlot => c.upload_slots,
                ResourceKind::UploadBytes => c.h2d_bytes,
                ResourceKind::InstallSlot => c.install_slots,
                ResourceKind::DeviceInstallBytes => c.device_install_bytes,
                ResourceKind::ResidentBytes => topology.resident_capacity_bytes(),
                ResourceKind::Waiter => waiters,
                _ => 4,
            };
            PhysicalResourceLimit::new(kind, capacity, 0)
        }))
        .unwrap();
        LoadRegistry::new(
            provider,
            broker,
            FairQueueConfig::for_production(limits).unwrap(),
        )
        .unwrap()
    }
    fn demand() -> ResourceDemand {
        ResourceDemand::required(ExecutionPhase::Decode)
    }
    fn waiter(id: u64) -> WaiterId {
        WaiterId::new(
            ferrule_common::execution::ExecutionTransactionId::new(id).unwrap(),
            RequestGeneration::new(1),
            DependencySetEpoch::new(1),
            ContinuationId::new(id),
        )
        .unwrap()
    }
    fn prepared(
        reg: &Registry,
        resolver: &mut RuntimeMaterializationResolver,
        request: MaterializationRequest,
    ) -> LoadRequest {
        let key = resolver
            .resolve(request)
            .expect("real pinned preparation through resolver");
        request.validate_key(key).unwrap();
        let load = reg
            .prepare_execution_request(key, demand(), ResourceRetention::ThroughStage)
            .unwrap();
        assert_eq!(load.preparation.key(), key);
        assert_eq!(
            load.preparation.binding().generation,
            key.destination_generation()
        );
        load
    }
    fn setup(
        fixture: &Fixture,
        ops: &CudaOperators,
        cleanup_fault: bool,
        waiters: u64,
    ) -> (Registry, RuntimeMaterializationResolver, Control) {
        let control = Control::new();
        let shared = fixture.provider(ops, true, cleanup_fault, control.clone());
        let resolver =
            RuntimeMaterializationResolver::new(shared.placement(), Some(shared.clone()));
        (registry(shared, waiters), resolver, control)
    }
    fn create(reg: &mut Registry, load: LoadRequest, id: u64) -> OperationId {
        let report = reg.attach_waiter(waiter(id), demand(), [load], 1).unwrap();
        assert_eq!(report.created.len(), 1);
        assert!(report.joined.is_empty());
        report.created[0]
    }
    fn next_action(reg: &mut Registry) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !reg.schedule_one(10).unwrap() {
            assert!(
                Instant::now() < deadline,
                "no feasible physical transition: {:?}",
                reg.stage_counts().collect::<Vec<_>>()
            );
        }
    }
    fn read_submitted(reg: &mut Registry, op: OperationId) {
        next_action(reg); // exact physical reserve, without submitting a read
        assert_eq!(reg.operation(op).unwrap().stage(), LoadStage::Reserved);
        next_action(reg);
        assert_eq!(reg.operation(op).unwrap().stage(), LoadStage::ReadSubmitted);
        assert_eq!(
            reg.pending_completions(),
            0,
            "a submit rejection is not a real read submission"
        );
    }
    fn collect_one(reg: &mut Registry) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while reg.collect_provider_completions(1) == 0 {
            assert!(
                Instant::now() < deadline,
                "physical completion deadline exceeded"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    fn complete_one(reg: &mut Registry) {
        collect_one(reg);
        reg.process_one_completion().unwrap();
    }
    fn no_publication(reg: &Registry, key: MaterializationKey) {
        assert_eq!(reg.stats().publications, 0);
        assert!(reg.residency_binding(key).is_none());
        assert_eq!(reg.resident_entries(), 0);
    }
    fn drain(reg: &mut Registry) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let report = reg
                .shutdown(100, 8)
                .expect("quiescent cancellation must drain");
            if report.drained {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "shutdown retained physical custody: {report:?}"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(reg.resources().active_grants(), 0);
        for kind in ResourceKind::ALL {
            assert_eq!(reg.resources().in_use(kind), 0, "{kind:?}");
        }
    }

    #[test]
    #[ignore = "opt-in GPU1; real provider rejection, no io_uring or FP4 success required"]
    fn positioned_reader_rejected_before_registry_admission() {
        let ops = gpu();
        let fixture = Fixture::new();
        let control = Control::new();
        let shared = fixture.provider(&ops, false, false, control.clone());
        let mut resolver =
            RuntimeMaterializationResolver::new(shared.placement(), Some(shared.clone()));
        let mut reg = registry(shared, 2);
        let error = resolver.resolve(fixture.request).unwrap_err();
        assert!(
            matches!(
                error,
                MaterializationResolveError::Provider {
                    source: FailureReason::StorageUnavailable,
                    ..
                }
            ),
            "{error:?}"
        );
        assert_eq!(resolver.stats().resolves, 1);
        assert_eq!(reg.stats().operations_created, 0);
        assert_eq!(control.stats().prepare_cancellations, 0);
        assert_eq!(control.stats().active_leases, 0);
        drain(&mut reg);
    }

    #[test]
    #[ignore = "opt-in GPU1 and real pinned io_uring; ENOSYS fails"]
    fn prepared_single_flight_create_cancel_before_submit() {
        let ops = gpu();
        let fixture = Fixture::new();
        let (mut reg, mut resolver, control) = setup(&fixture, &ops, false, 2);
        let load = prepared(&reg, &mut resolver, fixture.request);
        let op = create(&mut reg, load, 1);
        let repeated = prepared(&reg, &mut resolver, fixture.request);
        assert_eq!(repeated.key, load.key);
        assert_eq!(repeated.preparation.binding(), load.preparation.binding());
        let join = reg
            .attach_waiter(waiter(2), demand(), [repeated], 2)
            .unwrap();
        assert_eq!(join.joined, [op]);
        assert!(join.created.is_empty());
        assert_eq!(reg.resources().in_use(ResourceKind::ReadSlot), 0);
        reg.detach_waiter(waiter(1), CancellationReason::ExternalRequest, 3)
            .unwrap();
        assert!(!reg.operation(op).unwrap().cancellation_requested());
        reg.detach_waiter(waiter(2), CancellationReason::ExternalRequest, 4)
            .unwrap();
        drain(&mut reg);
        assert_eq!(reg.stats().operations_created, 1);
        assert_eq!(reg.stats().physical_completions, 0);
        assert_eq!(control.stats().prepare_cancellations, 1);
        no_publication(&reg, load.key);
    }

    #[test]
    #[ignore = "opt-in GPU1 and real pinned io_uring; ENOSYS fails"]
    fn admission_rejection_keeps_exact_cleanup_token_until_retry() {
        let ops = gpu();
        let fixture = Fixture::new();
        let (mut reg, mut resolver, control) = setup(&fixture, &ops, false, 0);
        let load = prepared(&reg, &mut resolver, fixture.request);
        assert!(reg.attach_waiter(waiter(1), demand(), [load], 1).is_err());
        assert_eq!(reg.stats().operations_created, 0);
        assert_eq!(reg.resources().active_grants(), 0);
        control.0.lock().unwrap().fail_cancel = 1;
        assert!(reg.discard_preparations([load.key]).is_err());
        assert_eq!(control.stats().prepare_cancellations, 0);
        reg.discard_preparations([load.key]).unwrap();
        let state = control.0.lock().unwrap();
        assert_eq!(state.attempts.len(), 2);
        assert_eq!(state.attempts[0], state.attempts[1]);
        drop(state);
        assert_eq!(control.stats().prepare_cancellations, 1);
        assert!(reg.provider().preparation(load.key).is_err());
        no_publication(&reg, load.key);
        drain(&mut reg);
    }

    #[test]
    #[ignore = "opt-in GPU1 and real pinned io_uring; ENOSYS fails"]
    fn read_submitted_cancel_holds_credit_until_real_cqe_is_processed() {
        let ops = gpu();
        let fixture = Fixture::new();
        let (mut reg, mut resolver, control) = setup(&fixture, &ops, false, 1);
        let load = prepared(&reg, &mut resolver, fixture.request);
        let op = create(&mut reg, load, 1);
        read_submitted(&mut reg, op);
        let held = [
            ResourceKind::ReadSlot,
            ResourceKind::StorageReadBytes,
            ResourceKind::PinnedHostBytes,
        ]
        .map(|k| (k, reg.resources().in_use(k)));
        assert_eq!(held[0].1, load.plan.requirements.read_slots);
        assert_eq!(held[1].1, load.plan.requirements.storage_read_bytes);
        assert_eq!(held[2].1, load.plan.requirements.pinned_host_bytes);
        reg.detach_waiter(waiter(1), CancellationReason::ExternalRequest, 11)
            .unwrap();
        for (kind, bytes) in held {
            assert_eq!(reg.resources().in_use(kind), bytes);
        }
        collect_one(&mut reg);
        for (kind, bytes) in held {
            assert_eq!(
                reg.resources().in_use(kind),
                bytes,
                "polling alone cannot return credit"
            );
        }
        reg.process_one_completion().unwrap();
        drain(&mut reg);
        assert_eq!(control.stats().prepare_cancellations, 1);
        no_publication(&reg, load.key);
        assert!(
            !reg.stage_history(op)
                .unwrap()
                .contains(&LoadStage::UploadSubmitted)
        );
    }

    fn unsupported_upload(cleanup_fault: bool) {
        let ops = gpu();
        require_fp4(false);
        let fixture = Fixture::new();
        let (mut reg, mut resolver, control) = setup(&fixture, &ops, cleanup_fault, 1);
        let load = prepared(&reg, &mut resolver, fixture.request);
        let op = create(&mut reg, load, 1);
        read_submitted(&mut reg, op);
        complete_one(&mut reg);
        assert_eq!(reg.operation(op).unwrap().stage(), LoadStage::HostReady);
        next_action(&mut reg);
        assert_eq!(
            reg.operation(op).unwrap().stage(),
            LoadStage::UploadSubmitted
        );
        let kinds = [
            ResourceKind::PinnedHostBytes,
            ResourceKind::UploadSlot,
            ResourceKind::UploadBytes,
            ResourceKind::DeviceInstallBytes,
            ResourceKind::ResidentBytes,
        ];
        let held = kinds.map(|k| (k, reg.resources().in_use(k)));
        assert_eq!(held[0].1, load.plan.requirements.pinned_host_bytes);
        assert_eq!(held[1].1, 1);
        assert_eq!(held[2].1, BYTES);
        assert_eq!(
            held[3].1, 0,
            "install bytes are not admitted before Installing"
        );
        assert_eq!(held[4].1, load.plan.resident_bytes);
        collect_one(&mut reg);
        assert!(
            reg.process_one_completion().is_err(),
            "Unknown must not be a successful terminal completion"
        );
        let faults = reg.provider_faults();
        assert_eq!(faults.len(), 1);
        assert_eq!(faults[0].quiescence, QuiescenceEvidence::Unknown);
        assert_eq!(
            faults[0].scope,
            Some(
                CompletionExpectation::new(op, load.key, LoadStage::UploadSubmitted, BYTES)
                    .unwrap()
            )
        );
        let message = format!("{:?}", faults[0].failure);
        assert!(
            message.contains("private layout preparation failed"),
            "{message}"
        );
        if cleanup_fault {
            assert!(
                message.contains("deterministic failpoint: stream-wide sync"),
                "{message}"
            );
        }
        for _ in 0..3 {
            assert!(reg.shutdown(100, 8).is_err());
            for (kind, bytes) in held {
                assert_eq!(
                    reg.resources().in_use(kind),
                    bytes,
                    "unknown custody {kind:?}"
                );
            }
            assert_eq!(
                reg.operation(op).unwrap().stage(),
                LoadStage::UploadSubmitted
            );
            assert!(reg.retirement(op).is_none());
            no_publication(&reg, load.key);
        }
        assert_eq!(reg.resources().in_use(ResourceKind::InstallSlot), 0);
        assert_eq!(control.stats().installs, 0);
        assert_eq!(control.stats().prepare_cancellations, 0);
        eprintln!("real provider Unknown retains {held:?}; full FP4 success remains unsupported");
        // Do not drop the sole logical credit owner while the provider is quarantined.
        // One bounded fixture per opt-in test process; no invented completion to drain it.
        std::mem::forget(reg);
    }
    #[test]
    #[ignore = "opt-in GPU1, unsupported FP4 build and real pinned io_uring; ENOSYS fails"]
    fn unsupported_upload_unknown_retains_registry_credits() {
        unsupported_upload(false);
    }
    #[test]
    #[ignore = "opt-in GPU1, unsupported FP4 build and real pinned io_uring; ENOSYS fails"]
    fn unsupported_upload_cleanup_failure_retains_registry_credits() {
        unsupported_upload(true);
    }

    #[test]
    #[ignore = "opt-in GPU1 with SM103 FP4 and pinned io_uring; unsupported environment fails"]
    fn upload_submitted_cancel_waits_for_real_event_without_publication() {
        let ops = gpu();
        require_fp4(true);
        let fixture = Fixture::new();
        let (mut reg, mut resolver, control) = setup(&fixture, &ops, false, 1);
        let load = prepared(&reg, &mut resolver, fixture.request);
        let op = create(&mut reg, load, 1);
        read_submitted(&mut reg, op);
        complete_one(&mut reg);
        next_action(&mut reg);
        assert_eq!(
            reg.operation(op).unwrap().stage(),
            LoadStage::UploadSubmitted
        );
        let held = reg.resources().in_use(ResourceKind::UploadBytes);
        assert_eq!(held, BYTES);
        reg.detach_waiter(waiter(1), CancellationReason::ExternalRequest, 11)
            .unwrap();
        assert_eq!(reg.resources().in_use(ResourceKind::UploadBytes), held);
        collect_one(&mut reg);
        assert_eq!(reg.resources().in_use(ResourceKind::UploadBytes), held);
        reg.process_one_completion().unwrap();
        drain(&mut reg);
        no_publication(&reg, load.key);
        assert_eq!(control.stats().installs, 0);
        assert_eq!(control.stats().prepare_cancellations, 1);
    }

    #[test]
    #[ignore = "opt-in GPU1 with SM103 FP4 and pinned io_uring; unsupported environment fails"]
    fn full_provider_publication_follows_install_event_and_returns_transient_credit() {
        let ops = gpu();
        require_fp4(true);
        let fixture = Fixture::new();
        let (mut reg, mut resolver, control) = setup(&fixture, &ops, false, 2);
        let load = prepared(&reg, &mut resolver, fixture.request);
        let op = create(&mut reg, load, 1);
        let same = prepared(&reg, &mut resolver, fixture.request);
        let join = reg.attach_waiter(waiter(2), demand(), [same], 2).unwrap();
        assert_eq!(join.joined, [op]);
        read_submitted(&mut reg, op);
        complete_one(&mut reg);
        next_action(&mut reg);
        complete_one(&mut reg);
        assert_eq!(reg.operation(op).unwrap().stage(), LoadStage::Installing);
        next_action(&mut reg); // accept the install; real provider owns the event
        no_publication(&reg, load.key);
        assert_eq!(reg.resources().in_use(ResourceKind::InstallSlot), 1);
        collect_one(&mut reg); // native event has completed, registry has not applied it
        no_publication(&reg, load.key);
        assert_eq!(reg.resources().in_use(ResourceKind::InstallSlot), 1);
        assert_eq!(
            control.stats().installs,
            1,
            "real physical publication precedes registry publication"
        );
        reg.process_one_completion().unwrap();
        reg.drive(50, 8).unwrap();
        assert_eq!(
            reg.residency_binding(load.key),
            Some(load.preparation.binding())
        );
        assert_eq!(reg.stats().publications, 1);
        assert_eq!(reg.stats().operations_created, 1);
        for kind in [
            ResourceKind::ReadSlot,
            ResourceKind::StorageReadBytes,
            ResourceKind::PinnedHostBytes,
            ResourceKind::UploadSlot,
            ResourceKind::UploadBytes,
            ResourceKind::InstallSlot,
            ResourceKind::DeviceInstallBytes,
        ] {
            assert_eq!(reg.resources().in_use(kind), 0, "{kind:?}");
        }
        assert_eq!(
            reg.resources().in_use(ResourceKind::ResidentBytes),
            load.plan.resident_bytes
        );
        let ready = [
            reg.pop_ready(51).unwrap().unwrap(),
            reg.pop_ready(52).unwrap().unwrap(),
        ]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            ready,
            [ContinuationId::new(1), ContinuationId::new(2)]
                .into_iter()
                .collect()
        );
        assert_eq!(reg.pop_ready(53).unwrap(), None);
        drain(&mut reg);
        assert_eq!(control.stats().active_leases, 0);
    }

    #[test]
    #[ignore = "opt-in GPU1 with SM103 FP4 and pinned io_uring; unsupported environment fails"]
    fn install_submitted_cancel_keeps_credit_until_event_and_releases_execution_lease() {
        let ops = gpu();
        require_fp4(true);
        let fixture = Fixture::new();
        let (mut reg, mut resolver, control) = setup(&fixture, &ops, false, 1);
        let load = prepared(&reg, &mut resolver, fixture.request);
        let op = create(&mut reg, load, 1);
        read_submitted(&mut reg, op);
        complete_one(&mut reg);
        next_action(&mut reg);
        complete_one(&mut reg);
        next_action(&mut reg);
        assert_eq!(reg.operation(op).unwrap().stage(), LoadStage::Installing);
        // One provider poll activates and submits the real slot-table mutation;
        // it cannot also poll the newly queued install ticket in the same pass.
        assert_eq!(reg.collect_provider_completions(1), 0);
        assert_eq!(control.0.lock().unwrap().activations, 1);
        assert_eq!(control.stats().installs, 0);
        let held = [
            ResourceKind::InstallSlot,
            ResourceKind::DeviceInstallBytes,
            ResourceKind::ResidentBytes,
        ]
        .map(|kind| (kind, reg.resources().in_use(kind)));
        assert_eq!(held[0].1, 1);
        assert_eq!(held[1].1, load.plan.requirements.device_install_bytes);
        assert_eq!(held[2].1, load.plan.resident_bytes);
        reg.detach_waiter(waiter(1), CancellationReason::ExternalRequest, 11)
            .unwrap();
        assert!(reg.operation(op).unwrap().cancellation_requested());
        no_publication(&reg, load.key);
        for (kind, amount) in held {
            assert_eq!(reg.resources().in_use(kind), amount);
        }
        collect_one(&mut reg);
        no_publication(&reg, load.key);
        for (kind, amount) in held {
            assert_eq!(reg.resources().in_use(kind), amount);
        }
        // Already-submitted installation is not rolled back: the current provider
        // publishes its completed mapping and releases the cancelled execution lease.
        assert_eq!(control.stats().installs, 1);
        assert_eq!(control.stats().active_leases, 0);
        reg.process_one_completion().unwrap();
        reg.drive(50, 8).unwrap();
        assert_eq!(
            reg.residency_binding(load.key),
            Some(load.preparation.binding())
        );
        assert_eq!(reg.resources().in_use(ResourceKind::InstallSlot), 0);
        assert_eq!(reg.resources().in_use(ResourceKind::DeviceInstallBytes), 0);
        assert_eq!(
            reg.resources().in_use(ResourceKind::ResidentBytes),
            load.plan.resident_bytes
        );
        assert_eq!(
            reg.pop_ready(51).unwrap(),
            None,
            "cancelled waiter cannot resume"
        );
        drain(&mut reg);
    }

    #[test]
    #[ignore = "opt-in GPU1; BF16 adapter boundary only, NOT full provider or registry success"]
    fn bf16_real_h2d_event_boundary_only() {
        let ops = gpu();
        let shape = CudaArtifactLinearShape::Bf16Bytes {
            out_features: 32,
            in_features: 32,
        };
        let source = ops.pin_u8_host_buffer(&vec![0x3f; 2048]).unwrap();
        let mut frame = ops.allocate_artifact_linear_device(shape).unwrap();
        ops.reset_counters();
        let ticket = ops
            .overwrite_artifact_linear_from_pinned_async(&mut frame, shape, source.clone(), None)
            .unwrap();
        assert!(!source.is_uniquely_owned());
        ticket.synchronize().unwrap();
        assert!(ticket.is_complete().unwrap());
        assert_eq!(ops.counters().host_to_device_copies, 1);
        assert_eq!(ops.counters().host_to_device_bytes, 2048);
        assert_eq!(ops.counters().stream_wide_syncs, 0);
        drop(ticket);
        assert!(source.is_uniquely_owned());
        eprintln!(
            "BF16 H2D/event boundary passed; not CudaExpertMaterializationProvider or LoadRegistry success"
        );
    }
}
