use super::*;

#[test]
fn partial_root_startup_shuts_down_real_started_programs_and_attached_experts() {
    let fixture = Arc::new(fixture::Fixture::new());
    let probe = Arc::new(Probe::default());
    let spec = fixture.resources().spec().clone();
    let l = layout(&spec, 2, 2);
    let owner_layout = l.clone();
    let observed = probe.clone();
    let started = Arc::new(AtomicUsize::new(0));
    let started_owners = started.clone();
    let result = build(
        topology(2, 2),
        &spec,
        l,
        config(),
        options(),
        move |rank| {
            if rank.global == super::rank(3) {
                return Err(error("injected last root load failure"));
            }
            Ok(fixture.resources())
        },
        move |rank, stage| {
            started_owners.fetch_add(1, Ordering::Relaxed);
            let index = rank.global.get() as usize;
            Ok(stage.map_program(|inner| Observed {
                inner,
                parallel: owner_layout.stages()[index / 2].clone(),
                lane: index % 2,
                probe: observed,
            }))
        },
    );
    assert!(result.is_err());
    assert_eq!(started.load(Ordering::Relaxed), 3);
    assert_eq!(probe.stopped.load(Ordering::Relaxed), 3);
    assert_eq!(probe.expert_shutdowns.load(Ordering::Relaxed), 1);
}

#[test]
fn partial_expert_startup_failure_cleans_successful_sibling_roots() {
    let fixture = Arc::new(fixture::Fixture::new());
    let spec = fixture.resources().spec().clone();
    let l = layout(&spec, 2, 2);
    let owner_layout = l.clone();
    let probe = Arc::new(Probe::default());
    let observed = probe.clone();
    let started = Arc::new(AtomicUsize::new(0));
    let count = started.clone();
    let experts = Arc::new(AtomicUsize::new(0));
    let expert_count = experts.clone();
    let result = build(
        topology(2, 2),
        &spec,
        l,
        config(),
        options(),
        move |rank| {
            if rank.global.get() == 3 {
                EXPERT_STARTUP.with_borrow_mut(|slot| {
                    *slot = Some(Arc::new(move |expert| {
                        expert_count.fetch_add(1, Ordering::Relaxed);
                        if expert.slot.get() == 1 {
                            return Err(error("injected second EP owner startup failure"));
                        }
                        Ok(())
                    }))
                });
            }
            Ok(fixture.resources())
        },
        move |rank, stage| {
            count.fetch_add(1, Ordering::Relaxed);
            let index = rank.global.get() as usize;
            Ok(stage.map_program(|inner| Observed {
                inner,
                parallel: owner_layout.stages()[index / 2].clone(),
                lane: index % 2,
                probe: observed,
            }))
        },
    );
    assert!(result.is_err());
    assert_eq!(experts.load(Ordering::Relaxed), 2);
    assert_eq!(started.load(Ordering::Relaxed), 3);
    assert_eq!(probe.stopped.load(Ordering::Relaxed), 3);
    assert_eq!(probe.expert_shutdowns.load(Ordering::Relaxed), 1);
}

#[test]
fn dense_optional_groups_tp1_tp4_and_local_residency_admission() {
    for tp in [1, 4] {
        let fixture = Arc::new(fixture::Fixture::for_degree(4, true));
        let spec = fixture.resources().spec().clone();
        let l = layout(&spec, 2, tp);
        assert!(
            l.optional_expert_group(0, options().expert_limits)
                .unwrap()
                .is_none()
        );
        assert!(l.expert_group(0, options().expert_limits).is_err());
        let mut pp = PipelineParallelExecutor::new_hybrid_cpu_tensor(
            topology(2, tp),
            &spec,
            l,
            config(),
            options(),
            move |_| Ok(fixture.resources()),
        )
        .unwrap();
        pp.forward(SessionId(73), &[1, 2, 3], ForwardPhase::Prefill)
            .unwrap();
        assert!(
            pp.owner_stats()
                .unwrap()
                .iter()
                .all(|s| s.experts.is_empty())
        );
        pp.shutdown().unwrap();
    }
    let fixture = Arc::new(fixture::Fixture::new());
    let spec = fixture.resources().spec().clone();
    for experts in [false, true] {
        let mut limits = options();
        if experts {
            limits.max_expert_weight_bytes = 1;
        } else {
            limits.max_owner_resident_bytes = 1;
        }
        let fixture = fixture.clone();
        assert!(
            PipelineParallelExecutor::new_hybrid_cpu_tensor(
                topology(2, 2),
                &spec,
                layout(&spec, 2, 2),
                config(),
                limits,
                move |_| Ok(fixture.resources())
            )
            .is_err()
        );
    }
}

#[test]
fn declared_ep_degree_is_checked_only_on_the_explicit_reference_entry() {
    let fixture = Arc::new(fixture::Fixture::new());
    let spec = fixture.resources().spec().clone();
    let t = ValidatedParallelTopology::new(
        topology(2, 2).topology_id(),
        4,
        rank(0),
        ParallelismPlan::validated(1, 2, 2, 1, 1, 2).unwrap(),
    )
    .unwrap();
    let l = layout(&spec, 2, 2);
    let load = fixture.clone();
    let mut pp = PipelineParallelExecutor::new_hybrid_cpu_tensor(
        t,
        &spec,
        l,
        config(),
        options(),
        move |_| Ok(load.resources()),
    )
    .unwrap();
    pp.forward(SessionId(73), &[1, 2, 3], ForwardPhase::Prefill)
        .unwrap();
    assert_eq!(pp.execution_scopes().kv_participants().len(), 4);
    pp.shutdown().unwrap();
    let wrong = ValidatedParallelTopology::new(
        topology(2, 2).topology_id(),
        4,
        rank(0),
        ParallelismPlan::validated(1, 2, 4, 1, 1, 2).unwrap(),
    )
    .unwrap();
    assert!(
        PipelineParallelExecutor::new_hybrid_cpu_tensor(
            wrong,
            &spec,
            layout(&spec, 2, 2),
            config(),
            options(),
            |_| panic!("mismatch before startup")
        )
        .is_err()
    );
}

#[test]
fn explicit_reference_api_has_the_exact_default_compile_fail_signatures() {
    let _ = std::mem::size_of::<HybridCpuTensorConfig>();
    let _ = std::mem::size_of::<HybridCpuTensorPipelineStageProgram>();
    let _ = std::mem::size_of::<HybridCpuTensorStage>();
    let _ = PipelineParallelExecutor::new_hybrid_cpu_tensor::<
        fn(PipelineRank) -> ferrule_common::Result<BoundDecoderResources>,
    >;
}
