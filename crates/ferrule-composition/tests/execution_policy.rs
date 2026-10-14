//! CPU-only contract tests for the policy module before the catalog wires it in.
#[path = "../src/model_factory/execution_policy.rs"]
mod execution_policy;

use std::{ffi::OsStr, time::Duration};

use execution_policy::*;
use ferrule_common::ParallelismPlan;

const PIPES: ProcessSupport = ProcessSupport::Pipes {
    owner_capacity: 128,
};
const UNSHARDED: ExecutionCapabilities = ExecutionCapabilities {
    ranks: RankCapability::ThreadOrProcess,
    tensor: TensorCapability::Unsharded,
};
const TENSOR: ExecutionCapabilities = ExecutionCapabilities {
    ranks: RankCapability::ThreadOrProcess,
    tensor: TensorCapability::DeviceThreads { degrees: &[2, 4] },
};
const LOCAL: ExecutionCapabilities = ExecutionCapabilities {
    ranks: RankCapability::ThreadOnly,
    tensor: TensorCapability::Unsharded,
};
const AUTO: PlacementPolicy = PlacementPolicy {
    selection: DeviceSelection::AutomaticOrExplicit,
    sharing: DeviceSharing::Colocated,
};
const EXPLICIT_DISTINCT: PlacementPolicy = PlacementPolicy {
    selection: DeviceSelection::Explicit,
    sharing: DeviceSharing::Distinct,
};

fn plan(pp: usize, tp: usize, ep: usize) -> ParallelismPlan {
    ParallelismPlan {
        pipeline_parallel: pp,
        tensor_parallel: tp,
        expert_parallel: ep,
        ..Default::default()
    }
}
fn thread() -> TransportRequest<'static> {
    TransportRequest {
        ranks: RankTransport::Thread,
        launch_executable: None,
        timeout: None,
        restarts: None,
    }
}
fn process() -> TransportRequest<'static> {
    TransportRequest {
        ranks: RankTransport::Process,
        // Validation must not inspect or execute this path.
        launch_executable: Some(OsStr::new("/does-not-exist/policy-test-worker")),
        ..thread()
    }
}

#[test]
fn transport_retains_omission_and_explicit_defaults() {
    let omitted = validate_transport(thread(), ProcessSupport::Unavailable).unwrap();
    let explicit = validate_transport(
        TransportRequest {
            timeout: Some(DEFAULT_RANK_TIMEOUT),
            restarts: Some(0),
            ..thread()
        },
        ProcessSupport::Unavailable,
    )
    .unwrap();
    assert_eq!(omitted.timeout(), explicit.timeout());
    assert_eq!(omitted.timeout_ms(), None);
    assert_eq!(omitted.requested().timeout, None);
    assert_eq!(explicit.requested().timeout, Some(DEFAULT_RANK_TIMEOUT));
    assert_eq!(explicit.requested().restarts, Some(0));
    assert_ne!(omitted.requested(), explicit.requested());
    let child = validate_transport(process(), PIPES).unwrap();
    assert_eq!(child.timeout_ms(), Some(30000));
    assert_eq!(
        child.requested().launch_executable,
        process().launch_executable
    );
}

#[test]
fn transport_rejects_ignored_launches_restarts_and_custom_thread_deadlines() {
    for launch in [OsStr::new(""), OsStr::new("worker")] {
        assert_eq!(
            validate_transport(
                TransportRequest {
                    launch_executable: Some(launch),
                    ..thread()
                },
                PIPES
            ),
            Err(PolicyError::LaunchWithThreads)
        );
    }
    for ranks in [thread(), process()] {
        for restarts in [1, usize::MAX] {
            assert_eq!(
                validate_transport(
                    TransportRequest {
                        restarts: Some(restarts),
                        ..ranks
                    },
                    PIPES
                ),
                Err(PolicyError::RestartUnsupported)
            );
        }
    }
    for timeout in [
        Duration::ZERO,
        Duration::from_millis(1),
        Duration::from_secs(31),
    ] {
        assert_eq!(
            validate_transport(
                TransportRequest {
                    timeout: Some(timeout),
                    ..thread()
                },
                PIPES
            ),
            Err(PolicyError::ThreadDeadlineUnsupported)
        );
    }
}

#[test]
fn process_requires_platform_launch_and_exact_nonzero_milliseconds() {
    assert_eq!(
        validate_transport(process(), ProcessSupport::Unavailable),
        Err(PolicyError::ProcessUnavailable)
    );
    for launch_executable in [None, Some(OsStr::new(""))] {
        assert_eq!(
            validate_transport(
                TransportRequest {
                    launch_executable,
                    ..process()
                },
                PIPES
            ),
            Err(PolicyError::LaunchRequired)
        );
    }
    for timeout in [
        Duration::ZERO,
        Duration::from_nanos(1),
        Duration::from_nanos(1_000_001),
        Duration::MAX,
    ] {
        assert_eq!(
            validate_transport(
                TransportRequest {
                    timeout: Some(timeout),
                    ..process()
                },
                PIPES
            ),
            Err(PolicyError::DeadlineMilliseconds)
        );
    }
    for millis in [1, 30000, 60000] {
        let value = validate_transport(
            TransportRequest {
                timeout: Some(Duration::from_millis(millis)),
                ..process()
            },
            PIPES,
        )
        .unwrap();
        assert_eq!(value.timeout_ms(), Some(millis));
    }
}

#[test]
fn all_dimensions_reject_zero_and_replica_axes_stay_disabled() {
    for (name, axis) in ["DP", "TP", "EP", "SP", "CP", "PP"].into_iter().zip(0..6) {
        for value in [0, 2] {
            let mut p = plan(1, 1, 1);
            *match axis {
                0 => &mut p.data_parallel,
                1 => &mut p.tensor_parallel,
                2 => &mut p.expert_parallel,
                3 => &mut p.sequence_parallel,
                4 => &mut p.context_parallel,
                _ => &mut p.pipeline_parallel,
            } = value;
            if value == 0 {
                assert_eq!(
                    validate_topology(p, OwnerLayout::StageRoots),
                    Err(PolicyError::ZeroDimension(name))
                );
            } else if matches!(axis, 0 | 3 | 4) {
                assert_eq!(
                    validate_topology(p, OwnerLayout::StageRoots),
                    Err(PolicyError::UnsupportedDimension(name))
                );
            }
        }
    }
}

#[test]
fn owner_layouts_preserve_additive_attachments_and_root_colocation() {
    for (p, layout, logical, slots, processes) in [
        (plan(1, 1, 1), OwnerLayout::SingleOwner, 1, 1, 1),
        (plan(2, 4, 1), OwnerLayout::StageRoots, 8, 8, 8),
        (plan(1, 1, 4), OwnerLayout::RootWithExperts, 5, 4, 4),
        (
            plan(2, 1, 4),
            OwnerLayout::StageRootsWithExperts,
            10,
            10,
            10,
        ),
        (
            plan(2, 4, 2),
            OwnerLayout::StageRootsWithExperts,
            12,
            12,
            12,
        ),
    ] {
        let resolved = validate_topology(p, layout).unwrap();
        assert_eq!(resolved.parallelism(), p);
        assert_eq!(resolved.layout(), layout);
        assert_eq!(resolved.logical_owners(), logical);
        assert_eq!(resolved.device_slots(), slots);
        assert_eq!(resolved.process_owners(), processes);
    }
    for (p, layout) in [
        (plan(2, 1, 1), OwnerLayout::SingleOwner),
        (plan(1, 2, 1), OwnerLayout::SingleOwner),
        (plan(1, 1, 2), OwnerLayout::SingleOwner),
        (plan(2, 1, 2), OwnerLayout::StageRoots),
        (plan(2, 1, 2), OwnerLayout::RootWithExperts),
        (plan(1, 2, 2), OwnerLayout::RootWithExperts),
        (plan(1, 1, 1), OwnerLayout::RootWithExperts),
        (plan(2, 1, 1), OwnerLayout::StageRootsWithExperts),
    ] {
        assert_eq!(
            validate_topology(p, layout),
            Err(PolicyError::LayoutMismatch)
        );
    }
}

#[test]
fn owner_arithmetic_checks_addition_multiplication_and_rank_abi() {
    for (p, layout) in [
        (plan(usize::MAX, 2, 1), OwnerLayout::StageRoots),
        (plan(1, usize::MAX, 2), OwnerLayout::StageRootsWithExperts),
        (plan(usize::MAX, 1, 2), OwnerLayout::StageRootsWithExperts),
        (plan(1, 1, usize::MAX), OwnerLayout::RootWithExperts),
        (plan(u32::MAX as usize, 2, 1), OwnerLayout::StageRoots),
    ] {
        assert_eq!(
            validate_topology(p, layout),
            Err(PolicyError::OwnerOverflow)
        );
    }
    assert_eq!(
        validate_topology(plan(u32::MAX as usize, 1, 1), OwnerLayout::StageRoots)
            .unwrap()
            .logical_owners(),
        u32::MAX as usize
    );
}

#[test]
fn placement_preserves_explicit_defaults_without_allocating_automatic_lists() {
    let topology = validate_topology(plan(2, 1, 1), OwnerLayout::StageRoots).unwrap();
    let automatic = validate_placement(topology, PlacementTarget::Device, None, AUTO).unwrap();
    let devices = [0, 1];
    let explicit =
        validate_placement(topology, PlacementTarget::Device, Some(&devices), AUTO).unwrap();
    assert_eq!(automatic.requested_devices(), None);
    assert_eq!(explicit.requested_devices(), Some(devices.as_slice()));
    assert_eq!(explicit.policy(), AUTO);
    for slot in 0..3 {
        assert_eq!(automatic.device(slot), explicit.device(slot));
    }
    assert_eq!(automatic.device(2), None);
    let host = validate_placement(topology, PlacementTarget::Host, None, AUTO).unwrap();
    assert_eq!(host.device(0), None);
    for devices in [&[][..], &[0, 1][..]] {
        assert_eq!(
            validate_placement(topology, PlacementTarget::Host, Some(devices), AUTO),
            Err(PolicyError::DevicePlacementOnHost)
        );
    }
    assert_eq!(
        validate_placement(topology, PlacementTarget::Device, None, EXPLICIT_DISTINCT),
        Err(PolicyError::ExplicitDevicesRequired)
    );
}

#[test]
fn placement_enforces_selected_distinctness_count_and_ordinal_abi() {
    let topology = validate_topology(plan(2, 2, 2), OwnerLayout::StageRootsWithExperts).unwrap();
    let colocated = PlacementPolicy {
        sharing: DeviceSharing::Colocated,
        ..EXPLICIT_DISTINCT
    };
    let repeated = [0, 1, 0, 1, 2, 3, 2, 3];
    assert!(
        validate_placement(
            topology,
            PlacementTarget::Device,
            Some(&repeated),
            colocated
        )
        .is_ok()
    );
    assert_eq!(
        validate_placement(
            topology,
            PlacementTarget::Device,
            Some(&repeated),
            EXPLICIT_DISTINCT
        ),
        Err(PolicyError::DuplicateDevice)
    );
    assert_eq!(
        validate_placement(topology, PlacementTarget::Device, Some(&[]), colocated),
        Err(PolicyError::DeviceCount {
            expected: 8,
            actual: 0
        })
    );
    let single = validate_topology(plan(1, 1, 1), OwnerLayout::SingleOwner).unwrap();
    assert!(
        validate_placement(
            single,
            PlacementTarget::Device,
            Some(&[i32::MAX as usize]),
            AUTO
        )
        .is_ok()
    );
    assert_eq!(
        validate_placement(
            single,
            PlacementTarget::Device,
            Some(&[i32::MAX as usize + 1]),
            AUTO
        ),
        Err(PolicyError::DeviceOrdinal)
    );
    let too_many =
        validate_topology(plan(i32::MAX as usize + 1, 1, 1), OwnerLayout::StageRoots).unwrap();
    assert_eq!(
        validate_placement(too_many, PlacementTarget::Device, None, AUTO),
        Err(PolicyError::DeviceOrdinal)
    );
    let boundary =
        validate_topology(plan(i32::MAX as usize, 1, 1), OwnerLayout::StageRoots).unwrap();
    assert!(validate_placement(boundary, PlacementTarget::Device, None, AUTO).is_ok());
}

#[test]
fn tensor_capability_is_selected_not_inferred_from_layout_or_arithmetic() {
    for tp in [1, 2, 3, 4, 8] {
        for target in [PlacementTarget::Host, PlacementTarget::Device] {
            for ranks in [thread(), process()] {
                for selected in [UNSHARDED, TENSOR] {
                    for (ep, layout) in [
                        (1, OwnerLayout::StageRoots),
                        (2, OwnerLayout::StageRootsWithExperts),
                    ] {
                        let topology = validate_topology(plan(2, tp, ep), layout).unwrap();
                        let transport = validate_transport(ranks, PIPES).unwrap();
                        let expected = tp == 1
                            || (matches!(tp, 2 | 4)
                                && selected == TENSOR
                                && target == PlacementTarget::Device
                                && ranks.ranks == RankTransport::Thread);
                        assert_eq!(
                            validate_capabilities(topology, transport, target, selected).is_ok(),
                            expected,
                            "{tp} {ep} {target:?} {ranks:?} {selected:?}"
                        );
                    }
                }
            }
        }
    }
    let single = validate_topology(plan(1, 1, 1), OwnerLayout::SingleOwner).unwrap();
    assert_eq!(
        validate_capabilities(
            single,
            validate_transport(process(), PIPES).unwrap(),
            PlacementTarget::Device,
            LOCAL
        ),
        Err(PolicyError::ThreadRanksRequired)
    );
}

#[test]
fn process_ceiling_counts_children_not_shared_root_devices_as_extra_processes() {
    for (layout, pp, ep, count) in [
        (OwnerLayout::StageRoots, 128, 1, 128),
        (OwnerLayout::StageRootsWithExperts, 32, 3, 128),
        (OwnerLayout::RootWithExperts, 1, 128, 128),
    ] {
        let topology = validate_topology(plan(pp, 1, ep), layout).unwrap();
        assert_eq!(topology.process_owners(), count);
        let full = validate_transport(process(), PIPES).unwrap();
        assert!(validate_capabilities(topology, full, PlacementTarget::Device, UNSHARDED).is_ok());
        let short = validate_transport(
            process(),
            ProcessSupport::Pipes {
                owner_capacity: 127,
            },
        )
        .unwrap();
        assert_eq!(
            validate_capabilities(topology, short, PlacementTarget::Device, UNSHARDED),
            Err(PolicyError::ProcessCapacity)
        );
    }
}

#[test]
fn existing_local_and_root_expert_placement_cases_compose_without_family_policy() {
    let transport = validate_transport(thread(), PIPES).unwrap();
    let local = validate_topology(plan(1, 1, 1), OwnerLayout::SingleOwner).unwrap();
    validate_capabilities(local, transport, PlacementTarget::Device, LOCAL).unwrap();
    for devices in [None, Some(&[0][..]), Some(&[7][..])] {
        assert!(validate_placement(local, PlacementTarget::Device, devices, AUTO).is_ok());
    }
    for ep in [2, 4, 8] {
        let topology = validate_topology(plan(1, 1, ep), OwnerLayout::RootWithExperts).unwrap();
        let devices: Vec<_> = (4..4 + ep).collect();
        for ranks in [thread(), process()] {
            validate_capabilities(
                topology,
                validate_transport(ranks, PIPES).unwrap(),
                PlacementTarget::Device,
                UNSHARDED,
            )
            .unwrap();
        }
        let placement = validate_placement(
            topology,
            PlacementTarget::Device,
            Some(&devices),
            EXPLICIT_DISTINCT,
        )
        .unwrap();
        assert_eq!(placement.device(0), Some(4));
        assert_eq!(placement.device(ep), None);
        assert_eq!(
            validate_placement(topology, PlacementTarget::Device, None, EXPLICIT_DISTINCT),
            Err(PolicyError::ExplicitDevicesRequired)
        );
    }
}

#[test]
fn existing_attached_expert_shapes_need_no_per_pair_policy_function() {
    let transport = validate_transport(thread(), PIPES).unwrap();
    let selected = ExecutionCapabilities {
        ranks: RankCapability::ThreadOnly,
        ..TENSOR
    };
    let placement = PlacementPolicy {
        selection: DeviceSelection::Explicit,
        sharing: DeviceSharing::Colocated,
    };
    let policy = TopologyPolicy {
        layout: OwnerLayout::StageRootsWithExperts,
        execution: selected,
        placement,
    };
    assert_eq!(policy.layout, OwnerLayout::StageRootsWithExperts);
    assert_eq!(policy.execution, selected);
    assert_eq!(policy.placement, placement);
    for (pp, tp, ep) in [
        (2, 1, 2),
        (2, 1, 4),
        (2, 1, 8),
        (1, 2, 2),
        (1, 4, 2),
        (2, 2, 2),
        (2, 4, 2),
    ] {
        let topology =
            validate_topology(plan(pp, tp, ep), OwnerLayout::StageRootsWithExperts).unwrap();
        validate_capabilities(topology, transport, PlacementTarget::Device, selected).unwrap();
        let devices = vec![0; topology.device_slots()];
        validate_placement(topology, PlacementTarget::Device, Some(&devices), placement).unwrap();
    }
    // Structural success deliberately does not admit a new provider combination.
    // Its existing exact PP/EP/TP guards must run as well when this module is wired.
    let structural_only =
        validate_topology(plan(3, 2, 3), OwnerLayout::StageRootsWithExperts).unwrap();
    assert!(
        validate_capabilities(
            structural_only,
            transport,
            PlacementTarget::Device,
            selected
        )
        .is_ok()
    );
}

#[test]
fn explicit_option_policy_distinguishes_consumed_default_only_and_ignored() {
    for requested in [None, Some(&0), Some(&1)] {
        assert!(validate_override("cache", requested, &0, OverridePolicy::Consumed).is_ok());
        assert_eq!(
            validate_override("cache", requested, &0, OverridePolicy::DefaultOnly).is_ok(),
            requested != Some(&1)
        );
        assert_eq!(
            validate_override("cache", requested, &0, OverridePolicy::Unsupported).is_ok(),
            requested.is_none()
        );
    }
    assert_eq!(
        validate_override("cache", Some(&0), &0, OverridePolicy::Unsupported),
        Err(PolicyError::ExplicitOptionUnsupported("cache"))
    );
    assert_eq!(
        validate_override("cache", Some(&1), &0, OverridePolicy::DefaultOnly),
        Err(PolicyError::NondefaultOptionUnsupported("cache"))
    );
}

#[test]
fn policy_source_has_no_concrete_family_runtime_or_io_dependency() {
    let source = include_str!("../src/model_factory/execution_policy.rs");
    for forbidden in [
        "ModelFamily",
        "Qwen",
        "qwen",
        "DeepSeek",
        "ferrule_runtime::",
        "ferrule_model::",
        "ferrule_backend::",
        "super::",
        "std::fs",
        "std::process",
        "std::env",
        "Instant::now",
        "hybrid_tensor",
        "ArithmeticPrecision",
        "NumericFp8",
        "ModelPrecision",
    ] {
        assert!(
            !source.contains(forbidden),
            "pure policy acquired {forbidden}"
        );
    }
}

#[test]
fn common_default_validation_preserves_explicit_policy() {
    validate_defaults(&[("cache", true), ("prefix", true)]).unwrap();
    assert!(validate_defaults(&[("cache", false)]).is_err());
}
