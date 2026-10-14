use super::*;

use crate::build_process_child as numeric_child_artifact;

#[path = "expert_completion_wait.rs"]
mod completion_wait_tests;

fn numeric_child() -> PathBuf {
    static CHILD: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    CHILD
        .get_or_init(|| {
            numeric_child_artifact::build("ferrule-runtime", "example", "process_rank_child")
        })
        .clone()
}
fn numeric_boots(
    fixture: &Fixture,
) -> Vec<(
    crate::parallel::process::ProcessIdentity,
    crate::parallel::expert_process::NumericBoot,
)> {
    use crate::parallel::expert_process::*;
    use crate::parallel::process::decoder::{DecoderDevice, ExpertPlacementFrame};
    use crate::parallel::process::*;
    let members = vec![109, 42];
    let placement = ExpertPlacementFrame {
        source_scope: ferrule_common::topology::ExpertSourceScope::ExternalStage,
        source: 7,
        members: members.clone(),
        entries: (0..4).map(|e| (1, e, members[e % 2])).collect(),
        max_tokens: 4,
        max_bytes: 128,
        devices: vec![
            DecoderDevice::Cuda { ordinal: 0 },
            DecoderDevice::Cuda { ordinal: 1 },
        ],
        timeout_ms: 2000,
    };
    let group = placement.decode(1..2).unwrap();
    members
        .into_iter()
        .enumerate()
        .map(|(slot, owner)| {
            let identity = ProcessIdentity::new(
                ProcessGroupEpoch::new(7).unwrap(),
                ParallelRankId::new(owner),
                ProcessOwnerInstanceId::new(slot as u64 + 1).unwrap(),
            );
            (
                identity,
                NumericBoot {
                    numeric_expert_boot: 2,
                    numeric_profile:
                        ferrule_model::transformer::NumericFp8Precision::expert_wire_default(),
                    checkpoint: fixture.path.clone(),
                    total_layers: 2,
                    layers: 1..2,
                    placement: placement.clone(),
                    owner,
                    hidden: 8,
                    max_rows: 2,
                    max_parameter_bytes: 4096,
                    host_max_bytes: 8 << 20,
                    host_max_experts: 2,
                    host_workers: 1,
                    host_headroom_bytes: 0,
                    device_bytes: 4 << 20,
                    workspace_bytes: 1 << 20,
                    expected: owned_catalog(&fixture.resources, &group, ParallelRankId::new(owner))
                        .unwrap(),
                },
            )
        })
        .collect()
}
fn numeric_options() -> crate::parallel::process::ProcessOwnerConfig {
    crate::parallel::process::ProcessOwnerConfig {
        startup_timeout: Duration::from_millis(500),
        command_timeout: Duration::from_millis(500),
        terminate_grace: Duration::from_millis(20),
        kill_grace: Duration::from_millis(200),
        ..Default::default()
    }
}
fn numeric_buckets(
    boots: &[(
        crate::parallel::process::ProcessIdentity,
        crate::parallel::expert_process::NumericBoot,
    )],
) -> (ExpertDispatchContext, Vec<ExpertTokenBucket>) {
    let context = ExpertDispatchContext {
        transaction: ExecutionTransactionId::new(99).unwrap(),
        source_rank: ParallelRankId::new(7),
        layer: 1,
    };
    let buckets = boots
        .iter()
        .enumerate()
        .map(|(slot, (identity, boot))| ExpertTokenBucket {
            owner_rank: identity.rank,
            tokens: vec![ferrule_model::transformer::ExpertToken {
                transaction: context.transaction,
                sequence: 81 + slot as u64,
                source_rank: context.source_rank,
                source_row: slot,
                route_slot: 0,
                expert: ferrule_model::moe::ExpertId::new(1, boot.expected[0].expert),
                weight: -0.0,
                payload: vec![-0.0, 0.125, -2.0, 3.0, 0.0, 1.0, 7.0, 8.0],
            }],
        })
        .collect();
    (context, buckets)
}
struct NumericDirectory(PathBuf);
impl NumericDirectory {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "numeric-child-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for NumericDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn absent(pids: impl IntoIterator<Item = u32>) {
    for pid in pids {
        let deadline = Instant::now() + Duration::from_secs(3);
        while std::path::Path::new(&format!("/proc/{pid}")).exists() {
            assert!(Instant::now() < deadline, "child {pid} not reaped");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

#[test]
fn numeric_v2_scope_catalog_and_rank_local_warm_are_exact() {
    use crate::parallel::expert_process::*;
    use crate::parallel::frame_budget::NumericFrameSize;
    use crate::parallel::process::ProcessFrameLimits;
    use ferrule_model::transformer::host_experts::*;
    let fixture = Fixture::new(true);
    let boots = numeric_boots(&fixture);
    let mut union = BTreeSet::new();
    for (slot, (_, boot)) in boots.iter().enumerate() {
        assert_eq!(boot.rank().unwrap().slot.get() as usize, slot);
        boot.validate_resources(&fixture.resources).unwrap();
        let plan = HostExpertWarmPlan::for_owned(
            fixture.resources.state_dict(),
            boot.host_options(),
            4096,
            boot.expected.iter().map(|s| (s.layer, s.expert)),
        )
        .unwrap();
        assert_eq!(plan.experts(), 2);
        let bytes = plan.payload_bytes();
        let cache = plan.warm(|_| {}).unwrap();
        assert_eq!(cache.stats().io.bytes_read, bytes);
        assert_eq!(cache.stats().experts, 2);
        for source in &boot.expected {
            assert_eq!(source.layer, 1);
            assert!(union.insert((source.layer, source.expert)));
        }
        for field in 0..5 {
            let mut altered = boot.clone();
            match field {
                0 => altered.expected[0].identity[0] ^= 1,
                1 => altered.expected[0].content_hash[0] ^= 1,
                2 => altered.expected[0].encoding += 1,
                3 => altered.expected[0].generation += 1,
                _ => altered.expected[0].bytes += 1,
            }
            assert!(altered.validate_resources(&fixture.resources).is_err());
        }
    }
    assert_eq!(union, (0..4).map(|e| (1, e)).collect());
    for field in 0..7 {
        let mut boot = boots[0].1.clone();
        match field {
            0 => boot.numeric_expert_boot = 1,
            1 => boot.total_layers = 1,
            2 => boot.layers = 2..2,
            3 => boot.placement.source = 109,
            4 => boot.owner = 1,
            5 => boot.expected.push(boot.expected[0].clone()),
            _ => boot.placement.members[1] = 109,
        }
        assert!(boot.group().is_err());
    }
    let mut boundary = boots[0].1.clone();
    boundary.hidden = 16;
    boundary.placement.max_tokens = 2;
    let size = NumericFrameSize::new(boundary.hidden, boundary.placement.max_tokens).unwrap();
    let limits = ProcessFrameLimits {
        max_frame_bytes: size.frame_bytes(),
        max_command_bytes: size.payload_bytes(),
        max_output_bytes: size.payload_bytes(),
        max_config_bytes: 1,
        max_error_bytes: 1,
    };
    boundary.validate_frame_limits(limits).unwrap();
    for smaller in [
        ProcessFrameLimits {
            max_command_bytes: size.payload_bytes() - 1,
            ..limits
        },
        ProcessFrameLimits {
            max_output_bytes: size.payload_bytes() - 1,
            ..limits
        },
        ProcessFrameLimits {
            max_frame_bytes: size.frame_bytes() - 1,
            ..limits
        },
    ] {
        assert!(boundary.validate_frame_limits(smaller).is_err());
    }
    let mut oversized = boots[0].1.clone();
    oversized.hidden = usize::MAX;
    assert!(
        oversized
            .validate_frame_limits(crate::parallel::process::ProcessFrameLimits::default())
            .is_err()
    );
    let limits = crate::parallel::process::ProcessFrameLimits {
        max_output_bytes: 1,
        ..Default::default()
    };
    assert!(boots[0].1.validate_frame_limits(limits).is_err());
    let mut value = serde_json::to_value(&boots[0].1).unwrap();
    value.as_object_mut().unwrap().remove("total_layers");
    assert!(serde_json::from_value::<NumericBoot>(value).is_err());
    assert!(
        HostExpertWarmPlan::for_owned(
            fixture.resources.state_dict(),
            boots[0].1.host_options(),
            4096,
            [(1, 0), (1, 0)]
        )
        .is_err()
    );
    assert!(
        HostExpertWarmPlan::for_owned(
            fixture.resources.state_dict(),
            boots[0].1.host_options(),
            4096,
            [(99, 0)]
        )
        .is_err()
    );
}

#[test]
fn numeric_process_real_children_parallel_bits_stats_isolation_and_shutdown() {
    use crate::parallel::expert_process::*;
    use crate::parallel::process::*;
    let child = numeric_child();
    let fixture = Fixture::new(true);
    for isolation in [
        NumericProcessIsolation::Standalone,
        NumericProcessIsolation::Inherited,
    ] {
        let dir = NumericDirectory::new();
        let boots = numeric_boots(&fixture);
        let (context, buckets) = numeric_buckets(&boots);
        let launch = ProcessLaunch::new(&child)
            .arg("numeric_barrier")
            .env("FERRULE_NUMERIC_BARRIER", &dir.0);
        let mut executor = ExpertParallelExecutor::new_numeric_process_scoped(
            launch,
            boots,
            numeric_options(),
            isolation,
        )
        .unwrap();
        let stats = executor.process_stats().unwrap();
        let pids = stats.iter().map(|s| s.pid).collect::<Vec<_>>();
        assert_eq!(pids.iter().collect::<BTreeSet<_>>().len(), 2);
        assert!(pids.iter().all(|&p| p != std::process::id()));
        for &pid in &pids {
            let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap();
            let suffix = stat.rsplit_once(") ").unwrap().1;
            let pgrp: u32 = suffix.split_whitespace().nth(2).unwrap().parse().unwrap();
            if isolation == NumericProcessIsolation::Standalone {
                assert_eq!(pgrp, pid);
            } else {
                assert_ne!(pgrp, pid);
            }
        }
        let expected = buckets
            .iter()
            .map(|b| {
                b.tokens[0]
                    .payload
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let output = executor
            .workers
            .execute_batch(context, buckets, &mut |_| Ok(()))
            .unwrap();
        for (slot, (_, results)) in output.iter().enumerate() {
            assert_eq!(
                results[0]
                    .output
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected[slot]
            );
            assert_eq!(results[0].transaction, context.transaction);
        }
        assert_eq!(executor.outstanding(), 0);
        assert!(executor.owner_stats().is_err());
        assert!(
            executor
                .process_stats()
                .unwrap()
                .iter()
                .all(|s| s.calls == 1 && s.tokens == 1)
        );
        executor.shutdown().unwrap();
        absent(pids);
    }
}

#[test]
fn numeric_process_partial_startup_and_sibling_cleanup_preserve_typed_unknown() {
    use crate::parallel::process::*;
    let child = numeric_child();
    let fixture = Fixture::new(true);
    for mode in [
        "numeric_boot_hang",
        "numeric_boot_exit",
        "numeric_boot_unknown",
        "numeric_boot_hang_shutdown_hang",
        "numeric_boot_exit_shutdown_hang",
    ] {
        let dir = NumericDirectory::new();
        let launch = ProcessLaunch::new(&child)
            .arg(mode)
            .env("FERRULE_NUMERIC_BARRIER", &dir.0);
        let outcome = std::panic::catch_unwind(|| {
            ExpertParallelExecutor::new_numeric_process(
                launch,
                numeric_boots(&fixture),
                numeric_options(),
            )
        });
        assert_eq!(
            outcome.err().unwrap().downcast_ref(),
            Some(&PanicQuiescence::Unknown),
            "{mode}"
        );
        let pids = std::fs::read_dir(&dir.0)
            .unwrap()
            .map(|e| {
                std::fs::read_to_string(e.unwrap().path())
                    .unwrap()
                    .parse::<u32>()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(pids.len(), 2);
        absent(pids);
    }
    let invalid = ProcessLaunch::new(fixture.path.with_extension("nonexistent"));
    assert!(
        ExpertParallelExecutor::new_numeric_process(
            invalid,
            numeric_boots(&fixture),
            numeric_options()
        )
        .is_err()
    );
}

#[test]
fn numeric_process_invalid_reply_unknown_and_deadline_stop_all_children_no_replay() {
    use crate::parallel::process::*;
    let child = numeric_child();
    let fixture = Fixture::new(true);
    for mode in [
        "numeric_bad_reply",
        "numeric_nonfinite",
        "numeric_unknown",
        "numeric_hang",
        "numeric_exit",
    ] {
        let boots = numeric_boots(&fixture);
        let (context, buckets) = numeric_buckets(&boots);
        let mut executor = ExpertParallelExecutor::new_numeric_process(
            ProcessLaunch::new(&child).arg(mode),
            boots,
            numeric_options(),
        )
        .unwrap();
        let pids = executor
            .process_stats()
            .unwrap()
            .into_iter()
            .map(|s| s.pid)
            .collect::<Vec<_>>();
        if mode == "numeric_hang" {
            executor
                .workers
                .set_timeout(Duration::from_millis(30))
                .unwrap();
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor
                .workers
                .execute_batch(context, buckets.clone(), &mut |_| Ok(()))
        }));
        assert_eq!(
            result.unwrap_err().downcast_ref(),
            Some(&PanicQuiescence::Unknown)
        );
        assert!(executor.workers.unknown);
        assert!(executor.outstanding() > 0);
        assert!(
            executor
                .workers
                .execute_batch(context, buckets, &mut |_| Ok(()))
                .is_err()
        );
        absent(pids);
    }
}

#[test]
fn numeric_process_fenced_failure_drains_and_preserves_healthy_owner() {
    use crate::parallel::process::*;
    let child = numeric_child();
    let fixture = Fixture::new(true);
    let boots = numeric_boots(&fixture);
    let (context, buckets) = numeric_buckets(&boots);
    let mut executor = ExpertParallelExecutor::new_numeric_process(
        ProcessLaunch::new(child).arg("numeric_fenced"),
        boots,
        numeric_options(),
    )
    .unwrap();
    let pids = executor
        .process_stats()
        .unwrap()
        .into_iter()
        .map(|s| s.pid)
        .collect::<Vec<_>>();
    assert!(
        executor
            .workers
            .execute_batch(context, buckets, &mut |_| Ok(()))
            .is_err()
    );
    assert!(!executor.workers.unknown);
    assert_eq!(executor.outstanding(), 0);
    executor.shutdown().unwrap();
    absent(pids);
}

#[test]
fn batch_late_queued_completion_and_final_callback_keep_ep_tickets() {
    use std::cell::Cell;
    use std::rc::Rc;

    use dispatch_test::Phase;

    struct ResetDispatchHook;
    impl Drop for ResetDispatchHook {
        fn drop(&mut self) {
            dispatch_test::HOOK.with_borrow_mut(|hook| *hook = None);
            dispatch_test::NOW.with(|now| now.set(None));
        }
    }

    struct CompletedStatsWorker {
        inner: ExpertRankWorker,
        completed: std::sync::mpsc::SyncSender<()>,
    }
    impl ReplicaWorker<Command> for CompletedStatsWorker {
        type Output = Reply;
        type Error = Error;

        fn execute(&mut self, request: WorkRequest<Command>) -> Result<Reply> {
            assert!(matches!(request.input, Command::Stats));
            let reply = self.inner.execute(request)?;
            self.completed.send(()).unwrap();
            Ok(reply)
        }

        fn shutdown(&mut self) -> Result<()> {
            self.inner.shutdown()
        }
    }

    let fixture = Fixture::new(false);
    for expire_at in [Some(Phase::Poll), Some(Phase::BeforePublish), None] {
        let _reset = ResetDispatchHook;
        let start = Instant::now();
        let timeout = Duration::from_millis(100);
        dispatch_test::NOW.with(|now| {
            assert!(now.get().is_none());
            now.set(Some(start));
        });
        let group = Arc::new(group(2));
        let factory_group = Arc::clone(&group);
        let resources = Arc::clone(&fixture.resources);
        let (completed, completion) = std::sync::mpsc::sync_channel(1);
        let pool = DataParallelExecutor::new(
            DataParallelConfig {
                replicas: 2,
                max_outstanding_per_replica: 1,
                session_capacity: 2,
            },
            move |slot| {
                let rank = ExpertRank {
                    slot,
                    owner: factory_group.members[slot.get() as usize],
                };
                Ok(CompletedStatsWorker {
                    inner: ExpertRankWorker::prepare_cpu(&resources, rank, &factory_group, 4096)?,
                    completed,
                })
            },
        )
        .unwrap();
        let mut executor = ExpertParallelExecutor {
            workers: ExpertRankWorkers {
                group,
                pool: Some(pool),
                next_serial: 1,
                closed: false,
                unknown: false,
                timeout,
                tickets: BTreeMap::new(),
                processes: None,
            },
            #[cfg(feature = "cuda")]
            cuda: false,
        };
        let phase = Rc::new(Cell::new(None));
        let observed_phase = Rc::clone(&phase);
        dispatch_test::HOOK.with_borrow_mut(|hook| {
            assert!(hook.is_none());
            *hook = Some(Box::new(move |workers, phase| {
                if phase == Phase::Poll {
                    // Worker return alone does not prove enqueue. Wait for
                    // successful execution before cancellation, then join all
                    // owners without consuming the queued DP completion.
                    completion.recv_timeout(Duration::from_secs(30)).unwrap();
                    workers.pool().shutdown().unwrap();
                    assert!(workers.pool().shutdown_ready());
                    assert_eq!(workers.pool().outstanding(), 1);
                } else {
                    assert_eq!(workers.pool().outstanding(), 0);
                }
                assert_eq!(workers.outstanding(), 1);
                assert_eq!(
                    workers.tickets.values().next().unwrap().completed,
                    phase == Phase::BeforePublish
                );
                observed_phase.set(Some(phase));
            }));
        });
        let mut expired_in_callback = false;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor
                .workers
                .call_many(vec![(rank(0), Command::Stats)], &mut || {
                    // Advance time only inside the real callback, after the
                    // hook has verified DP credit and retained EP custody.
                    if expire_at.is_some() && phase.get() == expire_at {
                        dispatch_test::NOW.with(|now| now.set(Some(start + timeout)));
                        expired_in_callback = true;
                    }
                    Ok(())
                })
        }));
        if let Some(expired_phase) = expire_at {
            assert!(expired_in_callback);
            assert_eq!(phase.get(), Some(expired_phase));
            assert_eq!(
                result.unwrap_err().downcast_ref(),
                Some(&PanicQuiescence::Unknown)
            );
            assert!(executor.workers.unknown);
            assert!(executor.workers.closed);
            assert_eq!(executor.outstanding(), 1);
            assert_eq!(
                executor.workers.pool().outstanding(),
                usize::from(expired_phase == Phase::Poll)
            );
            assert_eq!(
                executor.workers.tickets.values().next().unwrap().completed,
                expired_phase == Phase::BeforePublish
            );
        } else {
            assert!(!expired_in_callback);
            assert_eq!(phase.get(), Some(Phase::BeforePublish));
            let replies = result.unwrap().unwrap();
            assert!(matches!(replies.as_slice(), [Reply::Stats(_)]));
            assert!(!executor.workers.unknown);
            assert_eq!(executor.workers.pool().outstanding(), 0);
            assert_eq!(executor.outstanding(), 0);
        }
        // These CPU owners were explicitly joined before the deadline test.
        // Dispose of the pool after the custody assertions, without leaking it
        // through the production unknown-quiescence Drop path.
        drop(executor.workers.pool.take());
    }
}

#[test]
fn process_shutdown_uses_external_deadline_not_fresh_local_budget() {
    use crate::parallel::process::*;
    let child = numeric_child();
    for mode in ["shutdown_hang", "drop_hang"] {
        let identity = ProcessIdentity::new(
            ProcessGroupEpoch::new(19).unwrap(),
            ParallelRankId::new(0),
            ProcessOwnerInstanceId::new(1).unwrap(),
        );
        let mut owner =
            ProcessRankOwner::<serde_json::Value, serde_json::Value, serde_json::Value>::spawn(
                ProcessLaunch::new(&child).arg(mode),
                identity,
                &serde_json::json!({}),
                ProcessOwnerConfig {
                    command_timeout: Duration::from_secs(30),
                    ..numeric_options()
                },
            )
            .unwrap();
        let pid = owner.pid().unwrap();
        let started = Instant::now();
        assert!(
            owner
                .shutdown_until(started + Duration::from_millis(50))
                .unwrap_err()
                .is_quiescence_unknown()
        );
        assert!(started.elapsed() < Duration::from_secs(2));
        absent([pid]);
    }
}

#[test]
fn expert_factory_clone_panic_is_guarded_on_owner_and_cleans_ready_siblings() {
    struct CloneBomb(Arc<AtomicUsize>);
    impl Clone for CloneBomb {
        fn clone(&self) -> Self {
            assert_eq!(
                self.0.fetch_add(1, Ordering::SeqCst),
                0,
                "injected factory clone panic"
            );
            Self(Arc::clone(&self.0))
        }
    }
    let fixture = Fixture::new(false);
    let resources = Arc::clone(&fixture.resources);
    let bomb = CloneBomb(Arc::new(AtomicUsize::new(0)));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ExpertRankWorkers::new(group(2), move |rank, group| {
            let _bomb = bomb;
            ExpertRankWorker::prepare_cpu(&resources, rank, group, 4096)
        })
    }));
    assert!(outcome.is_ok(), "user Clone escaped the DP initializer");
    assert!(outcome.unwrap().is_err());
}

#[cfg(all(unix, feature = "cuda"))]
use super::numeric_test_fixture as numeric_gpu_fixture;

#[cfg(all(unix, feature = "cuda"))]
#[test]
#[ignore = "real CUDA numeric child EP2/EP8 on devices 0..7; no fake protocol replies"]
fn numeric_process_gpu_ep2_ep8_owned_compute_oracle_and_cleanup() {
    use crate::parallel::expert_process::*;
    use crate::parallel::process::decoder::{DecoderDevice, ExpertPlacementFrame};
    use crate::parallel::process::*;
    let child = numeric_child();
    let dir = NumericDirectory::new();
    let compressed = dir.0.join("numeric.bin");
    let dense = dir.0.join("dense.bin");
    let resources = numeric_gpu_fixture::bind(&compressed, 8, true, true);
    let reference = numeric_gpu_fixture::bind(&dense, 8, false, true);
    let mut reports = Vec::new();
    for count in [2, 8] {
        let members = (0..count)
            .map(|slot| 109 - 7 * slot as u32)
            .collect::<Vec<_>>();
        let placement = ExpertPlacementFrame {
            source_scope: ferrule_common::topology::ExpertSourceScope::ExternalStage,
            source: 7,
            members: members.clone(),
            entries: (0..2)
                .flat_map(|layer| {
                    let members = &members;
                    (0..8).map(move |e| (layer, e, members[e % count]))
                })
                .collect(),
            max_tokens: 8,
            max_bytes: 8 * 8 * 4,
            devices: (0..count)
                .map(|ordinal| DecoderDevice::Cuda { ordinal })
                .collect(),
            timeout_ms: 30000,
        };
        let group = placement.decode(0..2).unwrap();
        let boots = members
            .iter()
            .enumerate()
            .map(|(slot, &owner)| {
                let id = ProcessIdentity::new(
                    ProcessGroupEpoch::new(count as u64).unwrap(),
                    ParallelRankId::new(owner),
                    ProcessOwnerInstanceId::new(slot as u64 + 1).unwrap(),
                );
                (
                    id,
                    NumericBoot {
                        numeric_expert_boot: NUMERIC_EXPERT_WIRE_VERSION,
                        numeric_profile:
                            ferrule_model::transformer::NumericFp8Precision::expert_wire_default(),
                        checkpoint: compressed.clone(),
                        total_layers: 2,
                        layers: 0..2,
                        placement: placement.clone(),
                        owner,
                        hidden: 8,
                        max_rows: 8,
                        max_parameter_bytes: 4096,
                        host_max_bytes: 8 << 20,
                        host_max_experts: 16 / count,
                        host_workers: 1,
                        host_headroom_bytes: 0,
                        device_bytes: 4 << 20,
                        workspace_bytes: 1 << 20,
                        expected: owned_catalog(&resources, &group, id.rank).unwrap(),
                    },
                )
            })
            .collect();
        let mut executor = ExpertParallelExecutor::new_numeric_process(
            ProcessLaunch::new(&child).arg("numeric_gpu"),
            boots,
            ProcessOwnerConfig {
                startup_timeout: Duration::from_secs(30),
                command_timeout: Duration::from_secs(30),
                ..Default::default()
            },
        )
        .unwrap();
        let before = executor.process_stats().unwrap();
        let pids = before.iter().map(|s| s.pid).collect::<Vec<_>>();
        assert_eq!(pids.iter().collect::<BTreeSet<_>>().len(), count);
        let mut union = BTreeSet::new();
        for s in &before {
            assert_ne!(s.pid, std::process::id());
            assert!(std::path::Path::new(&format!("/proc/{}", s.pid)).exists());
            assert_eq!(s.resident_experts, 16 / count);
            assert_eq!(s.resident_bytes, 16 / count * 294);
            assert_eq!(s.host_read_bytes as usize, s.resident_bytes);
            for expert in &s.owned_experts {
                assert!(union.insert(*expert));
            }
        }
        assert_eq!(union.len(), 16);
        let mut max_abs = 0.0f32;
        let mut compared = 0;
        for (iteration, layer) in [0, 1, 1, 0].into_iter().enumerate() {
            let context = ExpertDispatchContext {
                transaction: ExecutionTransactionId::new(99 + iteration as u64).unwrap(),
                source_rank: ParallelRankId::new(7),
                layer,
            };
            let buckets = members
                .iter()
                .map(|&owner| ExpertTokenBucket {
                    owner_rank: ParallelRankId::new(owner),
                    tokens: (0..8)
                        .filter(|&e| {
                            group.placement.owner(layer, e) == Some(ParallelRankId::new(owner))
                        })
                        .map(|e| ferrule_model::transformer::ExpertToken {
                            transaction: context.transaction,
                            sequence: 81,
                            source_rank: context.source_rank,
                            source_row: e,
                            route_slot: 0,
                            expert: ferrule_model::moe::ExpertId::new(layer, e),
                            weight: 1.0,
                            payload: (0..8)
                                .map(|i| (i as f32 - 3.25) * (e as f32 + 1.0) * 0.03)
                                .collect(),
                        })
                        .collect(),
                })
                .collect::<Vec<_>>();
            let expected = buckets
                .iter()
                .enumerate()
                .map(|(slot, b)| {
                    let mut worker = ExpertRankWorker::prepare_cpu(
                        &reference,
                        ExpertRank {
                            owner: b.owner_rank,
                            slot: ParallelRankId::new(slot as u32),
                        },
                        &group,
                        4096,
                    )
                    .unwrap();
                    worker.compute(&b.tokens).unwrap()
                })
                .collect::<Vec<_>>();
            let actual = executor
                .workers
                .execute_batch(context, buckets, &mut |id| {
                    assert_eq!(id, context.transaction);
                    Ok(())
                })
                .unwrap();
            for ((owner, results), expected) in actual.iter().zip(expected) {
                for (a, e) in results.iter().zip(expected) {
                    assert_eq!(a.owner_rank, *owner);
                    assert_eq!(a.transaction, context.transaction);
                    assert_eq!(a.expert, e.expert);
                    for (&a, &e) in a.output.iter().zip(&e.output) {
                        max_abs = max_abs.max((a - e).abs());
                        compared += 1;
                        assert!(
                            a.is_finite()
                                && e.is_finite()
                                && (a - e).abs() <= 2e-7 + 2e-5 * e.abs(),
                            "{a} != {e}"
                        );
                    }
                }
            }
            assert_eq!(executor.outstanding(), 0);
        }
        let after = executor.process_stats().unwrap();
        for (a, b) in before.iter().zip(&after) {
            assert_eq!(a.pid, b.pid);
            assert_eq!(a.owned_experts, b.owned_experts);
            assert_eq!(a.host_hits, b.host_hits);
            assert_eq!(a.host_read_bytes, b.host_read_bytes);
            assert_eq!(a.uploads, b.uploads);
            assert_eq!(b.evictions, 0);
            assert_eq!(b.pending_upload_bytes, 0);
            assert!(!b.quarantined);
            assert_eq!(b.calls, 4);
            assert_eq!(b.tokens, 32 / count);
        }
        let gpu = std::process::Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv",
            ])
            .output()
            .unwrap();
        assert!(gpu.status.success());
        executor.shutdown().unwrap();
        absent(pids);
        eprintln!(
            "real CUDA process EP{count}: {compared} values max_abs={max_abs:e}, children reaped"
        );
        reports.push(serde_json::json!({"ep":count,"values_compared":compared,"max_abs":max_abs,
            "before":before,"after":after,"gpu_live":String::from_utf8(gpu.stdout).unwrap(),"children_reaped":true}));
    }
    if let Some(output) = std::env::var_os("FERRULE_PROCESS_EP_TINY_REPORT") {
        std::fs::write(output, serde_json::to_vec_pretty(&reports).unwrap()).unwrap();
    }
}
