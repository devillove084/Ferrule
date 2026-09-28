use super::*;
use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use ferrule_model::nn::{ParameterDType, ParameterResidency, ParameterSpec};
use ferrule_model::transformer::{
    DecoderRecipe, ExactNameMapper, HostRows, NameMapping, RouterRoutes, RowsDType, RowsShape,
    StateDictBinder, StateDictSchema, SyntheticDecoderRecipe,
};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

struct Fixture {
    path: PathBuf,
    resources: Arc<BoundDecoderResources>,
}
impl Fixture {
    fn new(numeric: bool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "resident-ep-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let config = serde_json::json!({
            "vocab_size": 8, "hidden_size": 8, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 12,
            "num_experts": 4, "experts_per_token": 4, "max_position_embeddings": 16,
            "rms_norm_eps": 0.00001, "rope_theta": 10000.0, "tie_word_embeddings": false
        });
        let output = SyntheticDecoderRecipe::new().build(&config).unwrap();
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        let mut slices = Vec::new();
        let mut bytes = Vec::new();
        for p in output.schema().parameters() {
            let role = output.schema().role(p.id()).unwrap().clone();
            let compressed = numeric && matches!(p.residency(), ParameterResidency::Expert { .. });
            let parameter = if compressed {
                ParameterSpec::new(
                    p.id(),
                    p.path().clone(),
                    ParameterDType::F8E4M3,
                    p.shape().to_vec(),
                    p.residency().clone(),
                )
                .unwrap()
                .with_required_scale(ParameterDType::Bf16, [1, 1])
                .unwrap()
            } else {
                p.clone()
            };
            schema.register_with_role(parameter, role.clone()).unwrap();
            let name = p.path().to_string();
            mapper
                .insert(&name, NameMapping::weight(p.path().clone()))
                .unwrap();
            let offset = bytes.len();
            for i in 0..p.shape().iter().product::<usize>() {
                let index = (i * 7 + i / 5 + p.id().get() as usize) % 4;
                if compressed {
                    bytes.push([0x38, 0xb8, 0x30, 0xb0][index]);
                } else {
                    bytes.extend([0.125f32, -0.125, 0.0625, -0.0625][index].to_le_bytes());
                }
            }
            slices.push(CheckpointTensorSlice {
                name: name.clone(),
                role: role.clone(),
                path: path.clone(),
                offset: offset as u64,
                bytes: (bytes.len() - offset) as u64,
                shape: p.shape().to_vec(),
                dtype: if compressed {
                    CheckpointDType::F8E4M3
                } else {
                    CheckpointDType::F32
                },
            });
            if compressed {
                let scale = format!("{name}.scale");
                mapper
                    .insert(&scale, NameMapping::scale(p.path().clone()))
                    .unwrap();
                slices.push(CheckpointTensorSlice {
                    name: scale,
                    role,
                    path: path.clone(),
                    offset: bytes.len() as u64,
                    bytes: 2,
                    shape: vec![1, 1],
                    dtype: CheckpointDType::Bf16,
                });
                bytes.extend(0x3e00u16.to_le_bytes()); // exactly 0.125
            }
        }
        std::fs::write(&path, bytes).unwrap();
        let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
            .bind_slices(slices)
            .unwrap();
        Self {
            path,
            resources: Arc::new(
                BoundDecoderResources::new(output.spec().clone(), Arc::new(bound)).unwrap(),
            ),
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
fn rank(i: usize) -> ParallelRankId {
    ParallelRankId::new(109 - i as u32 * 7)
}
fn context() -> ExpertDispatchContext {
    ExpertDispatchContext {
        transaction: ExecutionTransactionId::new(9001).unwrap(),
        source_rank: rank(0),
        layer: 0,
    }
}
fn group(owners: usize) -> ExpertGroup {
    ExpertGroup {
        source_rank: rank(0),
        members: (0..owners).map(rank).collect(),
        layers: 0..2,
        placement: ExpertPlacement::new(
            (0..2).flat_map(|layer| (0..4).map(move |e| (layer, e, rank(e % owners)))),
        )
        .unwrap(),
        limits: ExpertDispatchLimits {
            max_tokens: 12,
            max_bytes: 12 * 8 * 4,
        },
    }
}
fn external_group(owners: usize) -> ExpertGroup {
    let mut group = group(owners);
    group.source_rank = ParallelRankId::new(0);
    group.members = (1..=owners)
        .map(|i| ParallelRankId::new(i as u32))
        .collect();
    group.placement = ExpertPlacement::new((0..2).flat_map(|layer| {
        (0..4).map(move |expert| {
            (
                layer,
                expert,
                ParallelRankId::new((expert % owners + 1) as u32),
            )
        })
    }))
    .unwrap();
    group
}

fn input() -> Rows {
    Rows::Host(
        HostRows::new(
            RowsShape::new(3, 8).unwrap(),
            RowsDType::F32,
            None,
            (0..24).map(|i| (i as f32 - 10.3) * 0.0327).collect(),
        )
        .unwrap(),
    )
}
fn routes() -> RouterRoutes {
    RouterRoutes::new(
        3,
        4,
        vec![3, 0, 2, 1, 1, 2, 0, 3, 2, 3, 1, 0],
        vec![0.4, 0.3, 0.2, 0.1, 0.3, 0.4, 0.1, 0.2, 0.2, 0.3, 0.4, 0.1],
    )
    .unwrap()
}
fn request<'a>(input: &'a Rows, routes: &'a RouterRoutes) -> RoutedSwiGluRequest<'a> {
    RoutedSwiGluRequest {
        context: context(),
        sequences: &[81, 81, 92],
        input,
        routes,
        arena: None,
    }
}
fn buckets(owners: usize) -> Vec<ExpertTokenBucket> {
    let g = group(owners);
    let p = ferrule_model::transformer::expert_parallel::ExpertDispatchPlan::new(
        context(),
        g.members,
        &g.placement,
        &routes(),
        &[81, 81, 92],
        8,
        g.limits,
    )
    .unwrap();
    p.dispatch(input().host().unwrap().values(), &mut |_| Ok(()))
        .unwrap()
}
fn cpu(fixture: &Fixture, owners: usize) -> ExpertParallelExecutor {
    let resources = fixture.resources.clone();
    ExpertParallelExecutor::new(group(owners), move |rank, group| {
        ExpertRankWorker::prepare_cpu(&resources, rank, group, 4096)
    })
    .unwrap()
}
fn ready(progress: OperatorProgress<Rows>) -> Rows {
    match progress {
        OperatorProgress::Ready(rows) => rows,
        _ => panic!("not ready"),
    }
}

struct Spy {
    entered: AtomicUsize,
    finished: Mutex<Vec<usize>>,
    owners: usize,
    fail: Option<usize>,
    unknown: bool,
    rendezvous: bool,
    delay: Duration,
}
struct ObservedWorker {
    inner: ExpertRankWorker,
    spy: Arc<Spy>,
}
impl ReplicaWorker<Command> for ObservedWorker {
    type Output = Reply;
    type Error = Error;
    fn execute(&mut self, request: WorkRequest<Command>) -> Result<Reply> {
        if matches!(request.input, Command::Compute { .. }) {
            let slot = self.inner.rank.slot.get() as usize;
            self.spy.entered.fetch_add(1, Ordering::SeqCst);
            let deadline = Instant::now() + Duration::from_secs(2);
            while self.spy.rendezvous && self.spy.entered.load(Ordering::SeqCst) < self.spy.owners {
                assert!(Instant::now() < deadline, "owners were dispatched serially");
                std::thread::sleep(Duration::from_micros(50));
            }
            std::thread::sleep(self.spy.delay * (self.spy.owners - slot) as u32);
            if self.spy.fail == Some(slot) {
                if self.spy.unknown {
                    std::panic::panic_any(PanicQuiescence::Unknown);
                }
                return Err(error("injected owner failure"));
            }
            self.spy.finished.lock().unwrap().push(slot);
        }
        self.inner.execute(request)
    }
    fn panic_quiescence(&mut self) -> PanicQuiescence {
        self.inner.panic_quiescence()
    }
    fn shutdown(&mut self) -> Result<()> {
        self.inner.shutdown()
    }
}
fn observed(fixture: &Fixture, spy: Arc<Spy>) -> ExpertParallelExecutor {
    let g = Arc::new(group(spy.owners));
    let resources = fixture.resources.clone();
    let factory_group = g.clone();
    let owners = spy.owners;
    let pool = DataParallelExecutor::new(
        DataParallelConfig {
            replicas: owners,
            max_outstanding_per_replica: 1,
            session_capacity: owners,
        },
        move |slot| {
            let rank = ExpertRank {
                slot,
                owner: factory_group.members[slot.get() as usize],
            };
            Ok(ObservedWorker {
                inner: ExpertRankWorker::prepare_cpu(&resources, rank, &factory_group, 4096)?,
                spy,
            })
        },
    )
    .unwrap();
    ExpertParallelExecutor {
        workers: ExpertRankWorkers {
            group: g,
            pool: Some(pool),
            next_serial: 1,
            closed: false,
            unknown: false,
            timeout: Duration::from_secs(5),
            tickets: BTreeMap::new(),
        },
        #[cfg(feature = "cuda")]
        cuda: false,
    }
}
fn spy(owners: usize) -> Spy {
    Spy {
        entered: AtomicUsize::new(0),
        finished: Mutex::new(Vec::new()),
        owners,
        fail: None,
        unknown: false,
        rendezvous: true,
        delay: Duration::from_millis(80),
    }
}

#[test]
fn two_and_four_owners_overlap_reverse_arrival_and_match_local_bitwise() {
    let fixture = Fixture::new(false);
    let input = input();
    let routes = routes();
    let expected = ready(
        cpu(&fixture, 1)
            .routed_swiglu(request(&input, &routes), &mut |_| Ok(()))
            .unwrap(),
    );
    for owners in [2, 4] {
        let spy = Arc::new(spy(owners));
        let mut executor = observed(&fixture, spy.clone());
        let start = Instant::now();
        let actual = ready(
            executor
                .routed_swiglu(request(&input, &routes), &mut |_| Ok(()))
                .unwrap(),
        );
        assert_eq!(
            actual.host().unwrap().values(),
            expected.host().unwrap().values()
        );
        assert_eq!(
            *spy.finished.lock().unwrap(),
            (0..owners).rev().collect::<Vec<_>>()
        );
        let elapsed = start.elapsed();
        if owners == 4 {
            assert!(
                elapsed < Duration::from_millis(700),
                "serial sum = 800 ms, actual {elapsed:?}"
            );
        }
        assert_eq!(executor.outstanding(), 0);
        for stats in executor.owner_stats().unwrap() {
            assert_eq!(stats.last_context, Some(context()));
            assert_eq!(stats.owned_experts.len(), 8 / owners);
        }
        executor.shutdown().unwrap();
    }
}

#[test]
fn partial_admission_cancels_drains_and_does_not_reuse_ticket_serials() {
    let fixture = Fixture::new(false);
    let mut state = spy(4);
    state.rendezvous = false;
    let mut executor = observed(&fixture, Arc::new(state));
    let mut checks = 0;
    let result = executor
        .workers
        .execute_batch(context(), buckets(4), &mut |id| {
            assert_eq!(id, context().transaction);
            checks += 1;
            if checks == 5 {
                Err(error("cancel partial admission"))
            } else {
                Ok(())
            }
        });
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("partial admission")
    );
    assert_eq!(executor.outstanding(), 0);
    assert!(executor.workers.next_serial > 1);
    executor
        .workers
        .execute_batch(context(), buckets(4), &mut |_| Ok(()))
        .unwrap();
    executor.shutdown().unwrap();
}

#[test]
fn failed_owner_drains_other_admitted_tickets_and_unknown_is_permanent() {
    let fixture = Fixture::new(false);
    for unknown in [false, true] {
        let mut state = spy(4);
        state.fail = Some(3);
        state.unknown = unknown;
        let mut executor = observed(&fixture, Arc::new(state));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor
                .workers
                .execute_batch(context(), buckets(4), &mut |_| Ok(()))
        }));
        if unknown {
            assert_eq!(
                result.unwrap_err().downcast_ref(),
                Some(&PanicQuiescence::Unknown)
            );
            assert!(executor.workers.unknown);
            assert!(executor.outstanding() > 0);
            std::thread::sleep(Duration::from_millis(400));
            assert!(executor.workers.unknown);
            assert!(
                executor
                    .workers
                    .execute_batch(context(), buckets(4), &mut |_| Ok(()))
                    .is_err()
            );
        } else {
            assert!(
                result
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("owner failure")
            );
            assert_eq!(executor.outstanding(), 0);
            executor.shutdown().unwrap();
        }
    }
}

#[test]
fn deadline_retains_tickets_and_rejects_duplicate_or_invalid_outer_routes() {
    let fixture = Fixture::new(false);
    let mut executor = observed(&fixture, Arc::new(spy(2)));
    let mut duplicate = buckets(2);
    duplicate.push(duplicate[0].clone());
    assert!(
        executor
            .workers
            .execute_batch(context(), duplicate, &mut |_| Ok(()))
            .is_err()
    );
    let mut invalid = buckets(2);
    invalid[1].tokens[0].source_rank = rank(7);
    assert!(
        executor
            .workers
            .execute_batch(context(), invalid, &mut |_| Ok(()))
            .is_err()
    );
    assert_eq!(executor.workers.next_serial, 1);
    executor
        .workers
        .set_timeout(Duration::from_millis(5))
        .unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        executor
            .workers
            .execute_batch(context(), buckets(2), &mut |_| Ok(()))
    }));
    assert_eq!(
        result.unwrap_err().downcast_ref(),
        Some(&PanicQuiescence::Unknown)
    );
    assert!(executor.workers.unknown);
    assert_eq!(executor.outstanding(), 2);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "tiny numeric MoE only; explicitly uses physical devices 1 and 2"]
fn numeric_resident_two_and_four_owners_devices_1_2_match_local_and_share_host_image() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_model::decoder::{HybridCudaExpertProgress, HybridCudaRoutedExecutor};
    use ferrule_model::execution::ExecutionPrecisionPolicy;
    use ferrule_model::transformer::host_experts::{HostExpertCacheOptions, HostExpertWarmPlan};
    use ferrule_model::transformer::{CudaStandardDecoderOperators, StandardDecoderOperators};
    use std::rc::Rc;
    let compressed = Fixture::new(true);
    let dense = Fixture::new(false);
    let cache = Arc::new(
        HostExpertWarmPlan::new(
            compressed.resources.state_dict(),
            HostExpertCacheOptions {
                max_experts: 8,
                max_bytes: 8 << 20,
                ..Default::default()
            },
            4096,
        )
        .unwrap()
        .warm(|_| {})
        .unwrap(),
    );
    assert_eq!(cache.stats().experts, 8);
    let config = NumericExpertConfig {
        max_parameter_bytes: 4096,
        max_device_bytes: 1 << 20,
        workspace_bytes: 64 << 10,
        dispatch_timeout: Duration::from_secs(10),
    };
    let host = input();
    let routes = routes();
    let mut local = cpu(&dense, 1);
    let expected = ready(
        local
            .routed_swiglu(request(&host, &routes), &mut |_| Ok(()))
            .unwrap(),
    );
    let ops = Rc::new(CudaOperators::new_on_device(1).unwrap());
    let mut staging =
        CudaStandardDecoderOperators::new(ops.clone(), ExecutionPrecisionPolicy::f32(), &[])
            .unwrap();
    let input = staging.bind_rows(input()).unwrap();
    for (devices, external) in [
        (vec![1, 2], false),
        (vec![1, 2], true),
        (vec![1, 2, 1, 2], false),
    ] {
        let owners = devices.len();
        let group = if external {
            external_group(owners)
        } else {
            group(owners)
        };
        let source = group.source_rank;
        let members = group.members.clone();
        let before_hits = cache.hits();
        let mut executor = ExpertParallelExecutor::new_numeric_f32(
            group,
            compressed.resources.clone(),
            cache.clone(),
            devices,
            config,
        )
        .unwrap();
        assert_eq!(cache.hits() - before_hits, 24); // each projection exactly once, no second NAS prewarm
        let initial = executor.owner_stats().unwrap();
        assert_eq!(
            initial.iter().map(|s| s.rank.owner).collect::<Vec<_>>(),
            members
        );
        if external {
            assert!(initial.iter().all(|s| s.rank.owner != source));
        }
        for stats in &initial {
            let cache = stats.cuda_cache.unwrap();
            assert_eq!(stats.owned_experts.len(), 8 / owners);
            assert_eq!(cache.resident_experts, 8 / owners);
            assert_eq!(cache.resident_bytes, 8 / owners * 3 * 98);
            assert_eq!(cache.evictions, 0);
            assert_eq!(cache.uploads, (8 / owners) as u64);
            assert_eq!(cache.pending_upload_bytes, 0);
        }
        let mut adapter = executor.into_hybrid_cuda(ops.clone()).unwrap();
        if external {
            for spoof in [members[0], ParallelRankId::new(999)] {
                let mut call = request(&input, &routes);
                call.context.source_rank = spoof;
                assert!(adapter.routed_swiglu(call, &mut |_| Ok(())).is_err());
            }
            assert_eq!(adapter.drain().unwrap(), HybridCudaExpertProgress::Complete);
            assert!(adapter.owner_stats().unwrap().iter().all(|s| s.calls == 0));
        }
        let mut old_uploads = initial
            .iter()
            .map(|s| s.cuda_cache.unwrap().uploads)
            .collect::<Vec<_>>();
        for (iteration, layer) in [0, 1, 0].into_iter().enumerate() {
            let mut call = request(&input, &routes);
            call.context.layer = layer;
            call.context.source_rank = source;
            let output = ready(
                adapter
                    .routed_swiglu(call, &mut |id| {
                        assert_eq!(id, context().transaction);
                        Ok(())
                    })
                    .unwrap(),
            );
            let actual = staging.download_rows(output).unwrap();
            let mut local_call = request(&host, &routes);
            local_call.context.layer = layer;
            let reference = ready(local.routed_swiglu(local_call, &mut |_| Ok(())).unwrap());
            for (&a, &e) in actual
                .values()
                .iter()
                .zip(reference.host().unwrap().values())
            {
                assert!(
                    a.is_finite() && (a - e).abs() < 2e-7 + 2e-5 * e.abs(),
                    "numeric {a}, local {e}"
                );
            }
            assert_eq!(adapter.drain().unwrap(), HybridCudaExpertProgress::Complete);
            let stats = adapter.owner_stats().unwrap();
            for (slot, stats) in stats.into_iter().enumerate() {
                assert_eq!(stats.rank.owner, members[slot]);
                assert_eq!(stats.calls, iteration + 1);
                assert_eq!(stats.last_context.unwrap().source_rank, source);
                let cache = stats.cuda_cache.unwrap();
                assert_eq!(cache.pending_upload_bytes, 0);
                assert_eq!(cache.uploads, old_uploads[slot], "request-time weight H2D");
                assert_eq!(cache.evictions, 0);
                assert_eq!(cache.resident_experts, 8 / owners);
                old_uploads[slot] = cache.uploads;
            }
        }
        assert!(expected.host().unwrap().values().iter().any(|v| *v != 0.0));
        // Same ordinal != same owner. A foreign context's activation must fail
        // before dispatch, while a root-shared Rc works above.
        let foreign = Rc::new(CudaOperators::new_on_device(1).unwrap());
        let mut foreign =
            CudaStandardDecoderOperators::new(foreign, ExecutionPrecisionPolicy::f32(), &[])
                .unwrap();
        let foreign_input = foreign.bind_rows(super::tests::input()).unwrap();
        let mut foreign_call = request(&foreign_input, &routes);
        foreign_call.context.source_rank = source;
        assert!(
            adapter
                .routed_swiglu(foreign_call, &mut |_| Ok(()))
                .is_err()
        );
        assert_eq!(adapter.executor.outstanding(), 0);
        assert_eq!(adapter.drain().unwrap(), HybridCudaExpertProgress::Complete);
        let deadline = Instant::now() + Duration::from_secs(10);
        while adapter.shutdown().unwrap() != HybridCudaExpertProgress::Complete {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(
            adapter.shutdown().unwrap(),
            HybridCudaExpertProgress::Complete
        );
        assert!(adapter.owner_stats().is_err()); // never fake closed-owner stats
        assert_eq!(cache.hits() - before_hits, 24);
    }
    // A different bound image cannot use this image's host proof, even with
    // identical parameter IDs/shapes/content.
    let other = Fixture::new(true);
    assert!(
        ExpertParallelExecutor::new_numeric_f32(
            group(2),
            other.resources.clone(),
            cache.clone(),
            vec![1, 2],
            config
        )
        .is_err()
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "tiny numeric owners with forced overlap and loss; devices 1 and 2 only"]
fn numeric_devices_1_2_overlap_and_hybrid_unknown_never_acknowledges() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_model::decoder::{HybridCudaExpertProgress, HybridCudaRoutedExecutor};
    use ferrule_model::execution::ExecutionPrecisionPolicy;
    use ferrule_model::transformer::host_experts::{HostExpertCacheOptions, HostExpertWarmPlan};
    use ferrule_model::transformer::{CudaStandardDecoderOperators, StandardDecoderOperators};
    use std::rc::Rc;
    let fixture = Fixture::new(true);
    let dense = Fixture::new(false);
    let cache = Arc::new(
        HostExpertWarmPlan::new(
            fixture.resources.state_dict(),
            HostExpertCacheOptions {
                max_experts: 8,
                max_bytes: 8 << 20,
                ..Default::default()
            },
            4096,
        )
        .unwrap()
        .warm(|_| {})
        .unwrap(),
    );
    let config = NumericExpertConfig {
        max_parameter_bytes: 4096,
        max_device_bytes: 1 << 20,
        workspace_bytes: 64 << 10,
        dispatch_timeout: Duration::from_secs(5),
    };
    let ops = Rc::new(CudaOperators::new_on_device(1).unwrap());
    let mut staging =
        CudaStandardDecoderOperators::new(ops.clone(), ExecutionPrecisionPolicy::f32(), &[])
            .unwrap();
    let device_input = staging.bind_rows(input()).unwrap();
    let routes = routes();
    let reference = ready(
        cpu(&dense, 1)
            .routed_swiglu(request(&input(), &routes), &mut |_| Ok(()))
            .unwrap(),
    );
    for unknown in [false, true] {
        let mut state = spy(2);
        state.unknown = unknown;
        if unknown {
            state.fail = Some(1);
        }
        let state = Arc::new(state);
        let observed = state.clone();
        let g = Arc::new(group(2));
        let owners = g.clone();
        let resources = fixture.resources.clone();
        let host = cache.clone();
        let pool = DataParallelExecutor::new(
            DataParallelConfig {
                replicas: 2,
                max_outstanding_per_replica: 1,
                session_capacity: 2,
            },
            move |slot| {
                let rank = ExpertRank {
                    slot,
                    owner: owners.members[slot.get() as usize],
                };
                let ops = Rc::new(CudaOperators::new_on_device(1 + slot.get() as usize)?);
                let inner = ExpertRankWorker::prepare_cuda_numeric_f32(
                    &resources, rank, &owners, &host, config, ops,
                )?;
                Ok(ObservedWorker { inner, spy: state })
            },
        )
        .unwrap();
        let executor = ExpertParallelExecutor {
            cuda: true,
            workers: ExpertRankWorkers {
                group: g,
                pool: Some(pool),
                next_serial: 1,
                closed: false,
                unknown: false,
                timeout: config.dispatch_timeout,
                tickets: BTreeMap::new(),
            },
        };
        let mut adapter = executor.into_hybrid_cuda(ops.clone()).unwrap();
        let result = adapter.routed_swiglu(request(&device_input, &routes), &mut |_| Ok(()));
        assert_eq!(observed.entered.load(Ordering::SeqCst), 2);
        if unknown {
            assert!(result.is_err());
            adapter.on_error(context().transaction);
            for _ in 0..3 {
                std::thread::sleep(Duration::from_millis(100));
                assert_eq!(adapter.drain().unwrap(), HybridCudaExpertProgress::Unknown);
                assert!(adapter.owner_stats().is_err());
                assert_eq!(
                    adapter.shutdown().unwrap(),
                    HybridCudaExpertProgress::Unknown
                );
                assert!(adapter.executor.outstanding() > 0);
            }
        } else {
            assert_eq!(*observed.finished.lock().unwrap(), vec![1, 0]);
            let actual = staging.download_rows(ready(result.unwrap())).unwrap();
            for (&a, &e) in actual
                .values()
                .iter()
                .zip(reference.host().unwrap().values())
            {
                assert!((a - e).abs() <= 2e-7 + 2e-5 * e.abs());
            }
            assert_eq!(adapter.drain().unwrap(), HybridCudaExpertProgress::Complete);
            let deadline = Instant::now() + Duration::from_secs(5);
            while adapter.shutdown().unwrap() != HybridCudaExpertProgress::Complete {
                assert!(Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
    // Mutating the captured source after Ready fails closed, even with every
    // expert resident. A failed owner cannot evict then demand-reload next call.
    let mut executor = ExpertParallelExecutor::new_numeric_f32(
        group(2),
        fixture.resources.clone(),
        cache,
        vec![1, 2],
        config,
    )
    .unwrap();
    let before = executor.owner_stats().unwrap();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&fixture.path)
        .unwrap();
    file.set_len(file.metadata().unwrap().len() + 1).unwrap();
    let mut adapter = executor.into_hybrid_cuda(ops).unwrap();
    for _ in 0..2 {
        assert!(
            adapter
                .routed_swiglu(request(&device_input, &routes), &mut |_| Ok(()))
                .is_err()
        );
        assert_eq!(adapter.drain().unwrap(), HybridCudaExpertProgress::Complete);
    }
    for (before, after) in before.iter().zip(adapter.executor.owner_stats().unwrap()) {
        assert_eq!(
            before.cuda_cache.unwrap().uploads,
            after.cuda_cache.unwrap().uploads
        );
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while adapter.shutdown().unwrap() != HybridCudaExpertProgress::Complete {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn admission_failure_drains_existing_ticket_and_live_cancel_drains_all_owners() {
    let fixture = Fixture::new(false);
    let mut state = spy(2);
    state.rendezvous = false;
    let mut executor = observed(&fixture, Arc::new(state));
    let result = executor.workers.call_many(
        vec![
            (rank(0), Command::Stats),
            (ParallelRankId::new(999), Command::Stats),
        ],
        &mut || Ok(()),
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("unknown expert owner")
    );
    assert_eq!(executor.outstanding(), 0);
    assert_eq!(executor.workers.pool().outstanding(), 0);
    executor.shutdown().unwrap();

    let state = Arc::new(spy(4));
    let mut executor = observed(&fixture, state.clone());
    let result = executor
        .workers
        .execute_batch(context(), buckets(4), &mut |id| {
            assert_eq!(id, context().transaction);
            if state.entered.load(Ordering::SeqCst) == 4 {
                Err(error("cancel all admitted owners"))
            } else {
                Ok(())
            }
        });
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("cancel all admitted owners")
    );
    assert_eq!(state.entered.load(Ordering::SeqCst), 4);
    assert_eq!(state.finished.lock().unwrap().len(), 4);
    assert_eq!(executor.outstanding(), 0);
    assert_eq!(executor.workers.pool().outstanding(), 0);
    assert!(!executor.workers.unknown);
    executor.shutdown().unwrap();
}

#[test]
fn external_root_is_not_a_worker_and_exact_source_group_and_transaction_still_gate_dispatch() {
    let fixture = Fixture::new(false);
    let input = input();
    let routes = routes();
    let expected = ready(
        cpu(&fixture, 1)
            .routed_swiglu(request(&input, &routes), &mut |_| Ok(()))
            .unwrap(),
    );
    for owners in [2, 4] {
        let group = external_group(owners);
        let resources = fixture.resources.clone();
        let mut executor = ExpertParallelExecutor::new(group.clone(), move |rank, group| {
            ExpertRankWorker::prepare_cpu(&resources, rank, group, 4096)
        })
        .unwrap();
        let initial = executor.owner_stats().unwrap();
        assert_eq!(initial.len(), owners);
        assert_eq!(
            initial.iter().map(|s| s.rank.owner).collect::<Vec<_>>(),
            group.members
        );
        let mut ctx = context();
        ctx.source_rank = group.source_rank;
        let plan = ferrule_model::transformer::expert_parallel::ExpertDispatchPlan::new(
            ctx,
            group.members.clone(),
            &group.placement,
            &routes,
            &[81, 81, 92],
            8,
            group.limits,
        )
        .unwrap();
        let buckets = plan
            .dispatch(input.host().unwrap().values(), &mut |_| Ok(()))
            .unwrap();
        for source in [group.members[0], ParallelRankId::new(999)] {
            let mut spoof = ctx;
            spoof.source_rank = source;
            // Even an empty batch has the same attached-source authority.
            assert!(
                executor
                    .workers
                    .execute_batch(spoof, vec![], &mut |_| Ok(()))
                    .is_err()
            );
            assert!(
                executor
                    .workers
                    .execute_batch(spoof, buckets.clone(), &mut |_| Ok(()))
                    .is_err()
            );
            let mut spoofed = buckets.clone();
            spoofed[0].tokens[0].source_rank = source;
            assert!(
                executor
                    .workers
                    .execute_batch(ctx, spoofed, &mut |_| Ok(()))
                    .is_err()
            );
            let mut call = request(&input, &routes);
            call.context = spoof;
            assert!(executor.routed_swiglu(call, &mut |_| Ok(())).is_err());
        }
        let mut wrong_txn = buckets.clone();
        wrong_txn[0].tokens[0].transaction = ExecutionTransactionId::new(9002).unwrap();
        assert!(
            executor
                .workers
                .execute_batch(ctx, wrong_txn, &mut |_| Ok(()))
                .is_err()
        );
        let mut wrong_layer = ctx;
        wrong_layer.layer = 2;
        assert!(
            executor
                .workers
                .execute_batch(wrong_layer, buckets.clone(), &mut |_| Ok(()))
                .is_err()
        );
        assert!(
            executor
                .workers
                .execute_batch(ctx, buckets, &mut |_| Err(error("unknown transaction")))
                .is_err()
        );
        assert!(executor.owner_stats().unwrap().iter().all(|s| s.calls == 0));
        let mut call = request(&input, &routes);
        call.context = ctx;
        let actual = ready(
            executor
                .routed_swiglu(call, &mut |id| {
                    assert_eq!(id, ctx.transaction);
                    Ok(())
                })
                .unwrap(),
        );
        assert_eq!(
            actual.host().unwrap().values(),
            expected.host().unwrap().values()
        );
        for stats in executor.owner_stats().unwrap() {
            assert_ne!(stats.rank.owner, group.source_rank);
            assert_eq!(stats.calls, 1);
            assert_eq!(stats.last_context, Some(ctx));
        }
        assert_eq!(executor.outstanding(), 0);
        executor.shutdown().unwrap();
    }
    for case in 0..5 {
        let mut group = external_group(2);
        match case {
            0 => group.members[1] = group.members[0],
            1 => {
                group.placement = ExpertPlacement::new((0..2).flat_map(|layer| {
                    (0..4).map(move |expert| (layer, expert, ParallelRankId::new(0)))
                }))
                .unwrap()
            }
            2 => group.limits.max_tokens = 0,
            3 => group.limits.max_bytes = 0,
            4 => group.members.clear(),
            _ => unreachable!(),
        }
        let resources = fixture.resources.clone();
        assert!(
            ExpertParallelExecutor::new(group, move |rank, group| {
                ExpertRankWorker::prepare_cpu(&resources, rank, group, 4096)
            })
            .is_err(),
            "invalid group case {case}"
        );
    }
}
