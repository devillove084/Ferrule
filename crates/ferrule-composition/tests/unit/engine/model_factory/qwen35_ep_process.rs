use crate::Result;
use crate::model_factory::parallel::budget::host_startup_options;
use crate::model_factory::parallel::experts::OwnerPlan;
use crate::model_factory::parallel::process::*;
use crate::model_factory::qwen35::Qwen35ResolvedAdapter;
use crate::model_factory::qwen35::ep::tests::Fixture;
use crate::model_factory::qwen35::ep_unsupported;
use crate::model_factory::qwen35::finish_owner_admission;
use crate::model_factory::*;
use ferrule_model::transformer::BoundDecoderResources;
use ferrule_model::transformer::host_experts::{
    HostCacheMemoryPlan, HostExpertCacheOptions, HostExpertWarmPlan, HostMemoryBudget,
    HostMemorySnapshot, StandardHostStartupMemory,
};
use ferrule_runtime::parallel::data::PanicQuiescence;
use ferrule_runtime::parallel::expert_process::{NumericBoot, NumericFrameSize, owned_catalog};
use ferrule_runtime::parallel::process::decoder::DecoderDevice;
use ferrule_runtime::parallel::process::{ProcessLaunch, ProcessOwnerConfig};
use std::collections::BTreeSet;
use std::time::Duration;

use crate::build_process_child as child_artifact;

fn placement() -> Qwen35ExpertPlacement {
    Qwen35ExpertPlacement::from_options(
        &PipelineBuildOptions {
            parallelism: ferrule_common::ParallelismPlan {
                expert_parallel: 2,
                ..Default::default()
            },
            rank_backend: PipelineRankBackend::Process,
            process_launch: Some(
                ProcessLaunch::new("unused-metadata-only-child").arg("__rank-worker"),
            ),
            devices: Some(vec![3, 1]),
            rank_timeout: Duration::from_millis(500),
            ..Default::default()
        },
        BackendSelection::Cuda,
    )
    .unwrap()
}
fn host_options(f: &Fixture) -> HostExpertCacheOptions {
    host_startup_options(
        &f.resources,
        HostExpertCacheOptions {
            workers: 1,
            memory_budget: HostMemoryBudget {
                headroom_bytes: 0,
                ..Default::default()
            },
            ..Default::default()
        },
        32,
        1 << 30,
    )
    .unwrap()
}
fn scoped(f: &Fixture, p: &Qwen35ExpertPlacement, layers: std::ops::Range<usize>) -> OwnerPlan {
    OwnerPlan::for_stage(
        &f.resources,
        p,
        4,
        layers,
        ParallelRankId::new(7),
        vec![ParallelRankId::new(109), ParallelRankId::new(42)],
    )
    .unwrap()
}
fn plan(f: &Fixture) -> Plan {
    let p = placement();
    new_plan(
        &f.request(),
        &f.resources,
        &p,
        &scoped(f, &p, 0..2),
        host_options(f),
    )
    .unwrap()
}
fn snapshot(bytes: u64) -> HostMemorySnapshot {
    HostMemorySnapshot {
        system_available_bytes: bytes,
        process_rss_bytes: 0,
        cgroup_limit_bytes: None,
        cgroup_current_bytes: None,
    }
}

#[test]
fn process_factory_scoped_v2_boots_preserve_catalog_membership_and_device_order() {
    let f = Fixture::new();
    let p = placement();
    let full = plan(&f);
    let mut union = BTreeSet::new();
    for layers in [0..1, 1..2] {
        let owners = scoped(&f, &p, layers.clone());
        let plan = new_plan(&f.request(), &f.resources, &p, &owners, host_options(&f)).unwrap();
        for (slot, (identity, boot)) in plan.boots.iter().enumerate() {
            assert_eq!(boot.numeric_expert_boot, 2);
            assert_eq!(boot.total_layers, 2);
            assert_eq!(boot.layers, layers);
            assert_eq!(boot.placement.source, 7);
            assert_eq!(boot.placement.members, [109, 42]);
            assert_eq!(boot.rank().unwrap().slot.get() as usize, slot);
            assert_eq!(boot.owner, identity.rank.get());
            assert_eq!(
                boot.device().unwrap(),
                DecoderDevice::Cuda {
                    ordinal: [3, 1][slot]
                }
            );
            assert_eq!(boot.placement.timeout_ms, 500);
            boot.validate_resources(&f.resources).unwrap();
            let expected = owned_catalog(&f.resources, &owners.group, identity.rank).unwrap();
            assert_eq!(boot.expected, expected);
            assert_eq!(boot.host_max_experts, 2);
            assert_eq!(boot.expected.iter().map(|s| s.bytes).sum::<u64>(), 588);
            assert_eq!(
                boot.device_bytes,
                588 + OWNER_WORKSPACE + owners.activation_bytes
            );
            for source in &boot.expected {
                assert!(union.insert((source.layer, source.expert)));
                assert!(full.boots[slot].1.expected.contains(source));
            }
        }
    }
    assert_eq!(union.len(), 8);
    assert!(f.resources.host_experts().is_none());
    assert_eq!(full.options.startup_timeout, Duration::from_millis(500));
    assert_eq!(full.options.command_timeout, Duration::from_millis(500));
    assert_eq!(full.options.frame_limits.max_frame_bytes, 8 << 20);
    assert_eq!(
        full.launch.args,
        [std::ffi::OsString::from("__rank-worker")]
    );
    assert_ne!(full.boots[0].0.epoch, plan(&f).boots[0].0.epoch);
    for range in [0..0, 2..3, std::ops::Range { start: 3, end: 2 }] {
        assert!(
            OwnerPlan::for_stage(
                &f.resources,
                &p,
                4,
                range,
                ParallelRankId::new(7),
                vec![ParallelRankId::new(109), ParallelRankId::new(42)]
            )
            .is_err()
        );
    }
}

#[test]
fn process_factory_global_host_cap_ipc_metadata_and_root_peak_are_admitted_together() {
    let f = Fixture::new();
    let p = placement();
    let owners = scoped(&f, &p, 0..2);
    let plan = plan(&f);
    let warm = plan
        .boots
        .iter()
        .map(|(_, b)| {
            HostExpertWarmPlan::for_owned(
                f.resources.state_dict(),
                b.host_options(),
                b.max_parameter_bytes,
                b.expected.iter().map(|s| (s.layer, s.expert)),
            )
            .unwrap()
            .memory_plan()
        })
        .collect::<Vec<_>>();
    let shards = HostCacheMemoryPlan::serial_partitions(&warm).unwrap();
    assert_eq!(plan.cache.payload_bytes, 8 * 294);
    assert!(plan.cache.retained_metadata_bytes > shards.retained_metadata_bytes);
    let base = StandardHostStartupMemory::for_resources(&f.resources, 32, 1 << 30).unwrap();
    assert_eq!(
        plan.budget.base_resident_bytes,
        base.resident_bytes().unwrap()
    );
    assert_eq!(plan.budget.base_temporary_bytes, base.temporary_bytes());
    assert_eq!(plan.budget.base_already_resident_bytes, 0);
    let admission = plan.budget.admit(plan.cache, snapshot(u64::MAX)).unwrap();
    assert!(admission.required_available_bytes > shards.peak_bytes().unwrap());
    assert!(
        plan.admit(snapshot(admission.required_available_bytes - 1))
            .is_err()
    );
    plan.admit(snapshot(admission.required_available_bytes))
        .unwrap();
    let mut constrained = snapshot(u64::MAX);
    constrained.cgroup_limit_bytes = Some(admission.required_available_bytes);
    constrained.cgroup_current_bytes = Some(1);
    assert!(plan.admit(constrained).is_err());
    let mut cap = host_options(&f);
    cap.max_bytes = warm.iter().map(|w| w.peak_bytes().unwrap()).max().unwrap();
    assert!(cap.max_bytes < shards.peak_bytes().unwrap());
    assert!(new_plan(&f.request(), &f.resources, &p, &owners, cap).is_err());
    cap = host_options(&f);
    cap.max_bytes = shards.peak_bytes().unwrap();
    assert!(
        new_plan(&f.request(), &f.resources, &p, &owners, cap).is_err(),
        "IPC and child metadata must fit the aggregate cap too"
    );
    cap.max_bytes = plan.cache.peak_bytes().unwrap();
    new_plan(&f.request(), &f.resources, &p, &owners, cap).unwrap();
    assert_eq!(
        plan.boots[0]
            .1
            .host_options()
            .memory_budget
            .base_resident_bytes,
        0
    );
    cap = host_options(&f);
    cap.max_experts = 4; // Each child fits; their union does not.
    assert!(new_plan(&f.request(), &f.resources, &p, &owners, cap).is_err());
    let mut overflow = plan;
    overflow.budget.base_temporary_bytes = u64::MAX;
    assert!(overflow.admit(snapshot(u64::MAX)).is_err());
}

#[test]
fn process_factory_rejects_untransportable_batches_and_preserves_full32_wire_budget() {
    let f = Fixture::new();
    let p = placement();
    let mut owners = scoped(&f, &p, 0..2);
    owners.group.limits.max_tokens = usize::MAX;
    assert!(new_plan(&f.request(), &f.resources, &p, &owners, host_options(&f)).is_err());
    let mut boot = plan(&f).boots.remove(0).1;
    boot.hidden = 2048;
    boot.placement.max_tokens = 32 * 8;
    let limits = fit_frames(
        p.process_owner_config().unwrap(),
        boot.hidden,
        boot.placement.max_tokens,
    )
    .unwrap()
    .frame_limits;
    boot.validate_frame_limits(limits).unwrap();
    boot.placement.max_tokens = 64 * 8;
    assert!(
        boot.validate_frame_limits(p.process_owner_config().unwrap().frame_limits)
            .is_err()
    );
}

fn child() -> std::path::PathBuf {
    static CHILD: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    CHILD
        .get_or_init(|| child_artifact::build("ferrule-runtime", "example", "process_rank_child"))
        .clone()
}
fn absent(pids: &[u32]) {
    for pid in pids {
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while std::path::Path::new(&format!("/proc/{pid}")).exists() {
            assert!(
                std::time::Instant::now() < deadline,
                "child {pid} not reaped"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

#[test]
fn process_factory_cpu_fixture_uses_real_children_without_root_warm_and_reaps() {
    let f = Fixture::new();
    let mut plan = plan(&f);
    plan.launch = ProcessLaunch::new(child()).arg("numeric_echo");
    // Planning has completed; the CPU protocol fixture never reads weights.
    // A hidden root/full-shard warm here would fail rather than silently pass.
    std::fs::remove_file(f.dir.join("weights.bin")).unwrap();
    let mut executor = plan.start().unwrap();
    assert!(f.resources.host_experts().is_none());
    let before = executor.process_stats().unwrap();
    let p = placement();
    let owners = scoped(&f, &p, 0..2);
    check_ready(&owners, &p, &before).unwrap();
    let mut wrong = before.clone();
    wrong[1].device = wrong[0].device;
    assert!(check_ready(&owners, &p, &wrong).is_err());
    wrong = before.clone();
    wrong[1].pid = wrong[0].pid;
    assert!(check_ready(&owners, &p, &wrong).is_err());
    assert_eq!(before.len(), 2);
    assert_ne!(before[0].pid, before[1].pid);
    assert!(before.iter().all(|s| s.pid != std::process::id()));
    let after = executor.process_stats().unwrap();
    ferrule_runtime::parallel::expert::ImmutableOwnedBaseline::processes(before.clone())
        .unwrap()
        .check_processes(&after)
        .unwrap();
    for field in 0..4 {
        let mut changed = after.clone();
        match field {
            0 => changed[0].pid += 1,
            1 => changed[0].host_read_bytes += 1,
            2 => changed[0].uploads += 1,
            _ => changed[0].quarantined = true,
        }
        assert!(
            ferrule_runtime::parallel::expert::ImmutableOwnedBaseline::processes(before.clone())
                .unwrap()
                .check_processes(&changed)
                .is_err()
        );
    }
    executor.shutdown().unwrap();
    absent(&before.iter().map(|s| s.pid).collect::<Vec<_>>());
}

#[test]
fn process_factory_partial_boot_keeps_unknown_typed_and_cleans_siblings() {
    let child = child();
    for mode in ["numeric_boot_exit", "numeric_boot_hang_shutdown_hang"] {
        let f = Fixture::new();
        let mut plan = plan(&f);
        plan.launch = ProcessLaunch::new(&child)
            .arg(mode)
            .env("FERRULE_NUMERIC_BARRIER", &f.dir);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| plan.start()));
        assert_eq!(
            outcome.err().unwrap().downcast_ref::<PanicQuiescence>(),
            Some(&PanicQuiescence::Unknown)
        );
        let pids = [109, 42].map(|owner| {
            std::fs::read_to_string(f.dir.join(format!("pid-{owner}")))
                .unwrap()
                .parse()
                .unwrap()
        });
        absent(&pids);
    }
}

#[test]
fn process_factory_unknown_cleanup_is_not_stringified() {
    let result = std::panic::catch_unwind(|| {
        finish_owner_admission::<()>(Err(ep_unsupported("admission rejected")), || {
            std::panic::panic_any(PanicQuiescence::Unknown)
        })
    });
    assert_eq!(
        result.err().unwrap().downcast_ref::<PanicQuiescence>(),
        Some(&PanicQuiescence::Unknown)
    );
}

#[test]
fn fit_frames_preserves_ceilings_and_admits_exact_envelope_boundary() {
    let size = NumericFrameSize::new(2048, 256).unwrap();
    let mut options = ProcessOwnerConfig::default();
    let unchanged = options.frame_limits;
    options.frame_limits.max_frame_bytes = size.frame_bytes();
    let fitted = fit_frames(options, 2048, 256).unwrap();
    assert_eq!(fitted.frame_limits.max_frame_bytes, size.frame_bytes());
    assert_eq!(fitted.frame_limits.max_command_bytes, size.payload_bytes());
    assert_eq!(fitted.frame_limits.max_output_bytes, size.payload_bytes());
    assert_eq!(
        fitted.frame_limits.max_config_bytes,
        unchanged.max_config_bytes
    );
    assert_eq!(
        fitted.frame_limits.max_error_bytes,
        unchanged.max_error_bytes
    );
    assert!(size.fits(fitted.frame_limits));

    let mut options = ProcessOwnerConfig::default();
    options.frame_limits.max_frame_bytes = size.frame_bytes() - 1;
    assert!(fit_frames(options, 2048, 256).is_err());
    let small = fit_frames(ProcessOwnerConfig::default(), 1, 1).unwrap();
    assert_eq!(small.frame_limits, unchanged);
    assert!(fit_frames(ProcessOwnerConfig::default(), usize::MAX, 1).is_err());
    assert!(fit_frames(ProcessOwnerConfig::default(), 1, usize::MAX).is_err());
    assert!(fit_frames(ProcessOwnerConfig::default(), 0, (usize::MAX - 256) / 384).is_err());
}

#[test]
fn numeric_profile_factory_propagates_to_every_boot_without_changing_transport_budget() {
    let fixture = Fixture::new();
    let placement = placement();
    let owners = scoped(&fixture, &placement, 0..2);
    for profile in [NumericFp8Profile::F32, NumericFp8Profile::W8A16Bf16] {
        let mut request = fixture.request();
        let Qwen35ResolvedAdapter::Moe { options, .. } =
            request.family_options.qwen35_mut().unwrap()
        else {
            panic!("numeric MoE fixture");
        };
        options.numeric_profile = profile;
        let plan = new_plan(
            &request,
            &fixture.resources,
            &placement,
            &owners,
            host_options(&fixture),
        )
        .unwrap();
        for (slot, (_, boot)) in plan.boots.iter().enumerate() {
            assert_eq!(boot.numeric_profile, profile.precision());
            assert_eq!(boot.workspace_bytes, OWNER_WORKSPACE);
            assert_eq!(
                boot.device_bytes,
                owners.weights[slot] + OWNER_WORKSPACE + owners.activation_bytes
            );
            let restored: NumericBoot =
                serde_json::from_slice(&serde_json::to_vec(boot).unwrap()).unwrap();
            assert_eq!(restored.numeric_profile, profile.precision());
        }
    }
}

fn new_plan(
    request: &ResolvedModelRequest,
    resources: &BoundDecoderResources,
    placement: &Qwen35ExpertPlacement,
    owners: &OwnerPlan,
    host: HostExpertCacheOptions,
) -> Result<Plan> {
    let profile = request
        .family_options
        .qwen35()
        .unwrap()
        .moe()
        .unwrap()
        .numeric_profile;
    Plan::new(request, resources, placement, owners, host, profile)
}
