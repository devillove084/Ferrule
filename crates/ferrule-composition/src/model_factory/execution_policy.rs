//! Pure structural policy for composition. These checks neither select a model
//! nor promise a compiled backend, checkpoint compatibility, or live admission.
//!
//! A provider selects owner layout, placement and execution capabilities. Exact
//! supported model degrees remain provider policy; owner-local resource and
//! `ProcessOwnerConfig` deadline validation must still run before construction.
use std::{ffi::OsStr, time::Duration};

use ferrule_common::ParallelismPlan;

pub(crate) const DEFAULT_RANK_TIMEOUT: Duration = Duration::from_secs(30);

type Result<T> = std::result::Result<T, PolicyError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PolicyError {
    ZeroDimension(&'static str),
    UnsupportedDimension(&'static str),
    LayoutMismatch,
    OwnerOverflow,
    RestartUnsupported,
    ThreadDeadlineUnsupported,
    ProcessUnavailable,
    LaunchRequired,
    LaunchWithThreads,
    DeadlineMilliseconds,
    ProcessCapacity,
    ThreadRanksRequired,
    TensorUnsupported,
    TensorRequiresDeviceThreads,
    DevicePlacementOnHost,
    ExplicitDevicesRequired,
    DeviceCount { expected: usize, actual: usize },
    DeviceOrdinal,
    DuplicateDevice,
    ExplicitOptionUnsupported(&'static str),
    NondefaultOptionUnsupported(&'static str),
}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension(d) => write!(f, "parallelism dimension {d} must be nonzero"),
            Self::UnsupportedDimension(d) => write!(f, "parallelism dimension {d} must be 1"),
            Self::LayoutMismatch => f.write_str("parallelism does not match selected owner layout"),
            Self::OwnerOverflow => f.write_str("owner count exceeds the rank ABI"),
            Self::RestartUnsupported => f.write_str("rank restart/replay is unsupported"),
            Self::ThreadDeadlineUnsupported => {
                f.write_str("thread ranks require the default cooperative timeout")
            }
            Self::ProcessUnavailable => {
                f.write_str("process pipes are unavailable on this platform")
            }
            Self::LaunchRequired => {
                f.write_str("process ranks require a nonempty launch executable")
            }
            Self::LaunchWithThreads => f.write_str("thread ranks do not consume a process launch"),
            Self::DeadlineMilliseconds => {
                f.write_str("process timeout must fit nonzero whole u64 milliseconds")
            }
            Self::ProcessCapacity => f.write_str("process owner count exceeds process capacity"),
            Self::ThreadRanksRequired => f.write_str("selected capability requires thread ranks"),
            Self::TensorUnsupported => {
                f.write_str("tensor degree is unsupported by selected capability")
            }
            Self::TensorRequiresDeviceThreads => {
                f.write_str("tensor partitioning requires device thread ranks")
            }
            Self::DevicePlacementOnHost => {
                f.write_str("host execution does not consume device placement")
            }
            Self::ExplicitDevicesRequired => {
                f.write_str("selected placement requires explicit devices")
            }
            Self::DeviceCount { expected, actual } => {
                write!(f, "expected {expected} device ordinals, received {actual}")
            }
            Self::DeviceOrdinal => {
                f.write_str("device placement exceeds the signed device ordinal ABI")
            }
            Self::DuplicateDevice => f.write_str("selected placement requires distinct devices"),
            Self::ExplicitOptionUnsupported(name) => write!(
                f,
                "explicit {name} is unsupported, including its default value"
            ),
            Self::NondefaultOptionUnsupported(name) => {
                write!(f, "{name} must retain its default value")
            }
        }
    }
}

impl std::error::Error for PolicyError {}

/// An explicit default is not automatically an omitted/unused option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OverridePolicy {
    Consumed,
    DefaultOnly,
    Unsupported,
}

pub(crate) fn validate_override<T: PartialEq>(
    name: &'static str,
    requested: Option<&T>,
    default: &T,
    policy: OverridePolicy,
) -> Result<()> {
    match (requested, policy) {
        (Some(_), OverridePolicy::Unsupported) => Err(PolicyError::ExplicitOptionUnsupported(name)),
        (Some(value), OverridePolicy::DefaultOnly) if value != default => {
            Err(PolicyError::NondefaultOptionUnsupported(name))
        }
        _ => Ok(()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RankTransport {
    Thread,
    Process,
}

/// Platform facts supplied by composition, not probed by this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProcessSupport {
    #[cfg_attr(unix, allow(dead_code, reason = "non-Unix platform capability"))]
    Unavailable,
    Pipes {
        owner_capacity: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TransportRequest<'a> {
    pub ranks: RankTransport,
    pub launch_executable: Option<&'a OsStr>,
    pub timeout: Option<Duration>,
    pub restarts: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ValidatedTransport<'a> {
    requested: TransportRequest<'a>,
    timeout: Duration,
    timeout_ms: Option<u64>,
    process_capacity: Option<usize>,
}

impl<'a> ValidatedTransport<'a> {
    pub(crate) const fn requested(self) -> TransportRequest<'a> {
        self.requested
    }
    pub(crate) const fn timeout(self) -> Duration {
        self.timeout
    }
    #[cfg(test)]
    pub(crate) const fn timeout_ms(self) -> Option<u64> {
        self.timeout_ms
    }
}

pub(crate) fn validate_transport(
    requested: TransportRequest<'_>,
    platform: ProcessSupport,
) -> Result<ValidatedTransport<'_>> {
    if requested.restarts.unwrap_or(0) != 0 {
        return Err(PolicyError::RestartUnsupported);
    }
    let timeout = requested.timeout.unwrap_or(DEFAULT_RANK_TIMEOUT);
    let (timeout_ms, process_capacity) = match requested.ranks {
        RankTransport::Thread => {
            if timeout != DEFAULT_RANK_TIMEOUT {
                return Err(PolicyError::ThreadDeadlineUnsupported);
            }
            if requested.launch_executable.is_some() {
                return Err(PolicyError::LaunchWithThreads);
            }
            (None, None)
        }
        RankTransport::Process => {
            let ProcessSupport::Pipes { owner_capacity } = platform else {
                return Err(PolicyError::ProcessUnavailable);
            };
            if requested.launch_executable.is_none_or(OsStr::is_empty) {
                return Err(PolicyError::LaunchRequired);
            }
            let millis = validate_process_timeout(timeout)?;
            (Some(millis), Some(owner_capacity))
        }
    };
    Ok(ValidatedTransport {
        requested,
        timeout,
        timeout_ms,
        process_capacity,
    })
}

/// Wire projection also uses this check without needing a launch executable.
pub(crate) fn validate_process_timeout(timeout: Duration) -> Result<u64> {
    let millis =
        u64::try_from(timeout.as_millis()).map_err(|_| PolicyError::DeadlineMilliseconds)?;
    if millis == 0 || Duration::from_millis(millis) != timeout {
        return Err(PolicyError::DeadlineMilliseconds);
    }
    Ok(millis)
}

/// Logical owners and device-list slots are deliberately different for a local
/// root with external experts. Layout does not select arithmetic or a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OwnerLayout {
    SingleOwner,
    /// PP-major TP roots; EP must be 1.
    StageRoots,
    /// Local root shares expert slot 0. The device list has EP slots; there are
    /// 1 + EP logical owners, and only the EP children use process slots.
    RootWithExperts,
    /// PP-major TP roots followed by stage-major EP children. EP must exceed 1;
    /// list length is PP * (TP + EP), never PP * TP * EP.
    StageRootsWithExperts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ValidatedTopology {
    parallelism: ParallelismPlan,
    layout: OwnerLayout,
    logical_owners: usize,
    device_slots: usize,
    process_owners: usize,
}

impl ValidatedTopology {
    pub(crate) const fn parallelism(self) -> ParallelismPlan {
        self.parallelism
    }
    pub(crate) const fn layout(self) -> OwnerLayout {
        self.layout
    }
    pub(crate) const fn logical_owners(self) -> usize {
        self.logical_owners
    }
    pub(crate) const fn device_slots(self) -> usize {
        self.device_slots
    }
    pub(crate) const fn process_owners(self) -> usize {
        self.process_owners
    }
}

pub(crate) fn validate_topology(
    p: ParallelismPlan,
    layout: OwnerLayout,
) -> Result<ValidatedTopology> {
    for (name, value) in [
        ("DP", p.data_parallel),
        ("TP", p.tensor_parallel),
        ("EP", p.expert_parallel),
        ("SP", p.sequence_parallel),
        ("CP", p.context_parallel),
        ("PP", p.pipeline_parallel),
    ] {
        if value == 0 {
            return Err(PolicyError::ZeroDimension(name));
        }
    }
    for (name, value) in [
        ("DP", p.data_parallel),
        ("SP", p.sequence_parallel),
        ("CP", p.context_parallel),
    ] {
        if value != 1 {
            return Err(PolicyError::UnsupportedDimension(name));
        }
    }
    let (owners, slots, processes) = match layout {
        OwnerLayout::SingleOwner if p == ParallelismPlan::default() => (1, 1, 1),
        OwnerLayout::StageRoots if p.expert_parallel == 1 => {
            let n = p
                .pipeline_parallel
                .checked_mul(p.tensor_parallel)
                .ok_or(PolicyError::OwnerOverflow)?;
            (n, n, n)
        }
        OwnerLayout::RootWithExperts
            if p.pipeline_parallel == 1 && p.tensor_parallel == 1 && p.expert_parallel > 1 =>
        {
            (
                p.expert_parallel
                    .checked_add(1)
                    .ok_or(PolicyError::OwnerOverflow)?,
                p.expert_parallel,
                p.expert_parallel,
            )
        }
        OwnerLayout::StageRootsWithExperts if p.expert_parallel > 1 => {
            let n = p
                .tensor_parallel
                .checked_add(p.expert_parallel)
                .and_then(|n| p.pipeline_parallel.checked_mul(n))
                .ok_or(PolicyError::OwnerOverflow)?;
            (n, n, n)
        }
        _ => return Err(PolicyError::LayoutMismatch),
    };
    u32::try_from(owners).map_err(|_| PolicyError::OwnerOverflow)?;
    Ok(ValidatedTopology {
        parallelism: p,
        layout,
        logical_owners: owners,
        device_slots: slots,
        process_owners: processes,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlacementTarget {
    Host,
    Device,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RankCapability {
    ThreadOnly,
    ThreadOrProcess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorCapability {
    Unsharded,
    /// Selected provider's degrees, not a global support table. TP=1 is always
    /// unsharded; non-unit degrees require device thread execution.
    DeviceThreads {
        degrees: &'static [usize],
    },
}

/// Orthogonal execution restrictions. Model-specific PP/EP degree sets and
/// checkpoint shape checks remain in the provider selecting these capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExecutionCapabilities {
    pub ranks: RankCapability,
    pub tensor: TensorCapability,
}

pub(crate) fn validate_capabilities(
    topology: ValidatedTopology,
    transport: ValidatedTransport<'_>,
    target: PlacementTarget,
    selected: ExecutionCapabilities,
) -> Result<()> {
    validate_execution_kind(topology, transport.requested.ranks, target, selected)?;
    if transport
        .process_capacity
        .is_some_and(|limit| topology.process_owners() > limit)
    {
        return Err(PolicyError::ProcessCapacity);
    }
    Ok(())
}

/// Execution eligibility precedes launch validation, preserving fail-closed
/// capability errors even when a caller has not supplied process launch details.
pub(crate) fn validate_execution_kind(
    topology: ValidatedTopology,
    ranks: RankTransport,
    target: PlacementTarget,
    selected: ExecutionCapabilities,
) -> Result<()> {
    if selected.ranks == RankCapability::ThreadOnly && ranks != RankTransport::Thread {
        return Err(PolicyError::ThreadRanksRequired);
    }
    let tp = topology.parallelism().tensor_parallel;
    if tp > 1 {
        match selected.tensor {
            TensorCapability::DeviceThreads { degrees } if degrees.contains(&tp) => {}
            _ => return Err(PolicyError::TensorUnsupported),
        }
        if target != PlacementTarget::Device || ranks != RankTransport::Thread {
            return Err(PolicyError::TensorRequiresDeviceThreads);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceSelection {
    AutomaticOrExplicit,
    Explicit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceSharing {
    Distinct,
    Colocated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacementPolicy {
    pub selection: DeviceSelection,
    pub sharing: DeviceSharing,
}

/// Frozen selection of independent owner, execution and placement contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TopologyPolicy {
    pub layout: OwnerLayout,
    pub execution: ExecutionCapabilities,
    pub placement: PlacementPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ValidatedPlacement<'a> {
    target: PlacementTarget,
    topology: ValidatedTopology,
    policy: PlacementPolicy,
    requested_devices: Option<&'a [usize]>,
}

impl<'a> ValidatedPlacement<'a> {
    pub(crate) const fn requested_devices(self) -> Option<&'a [usize]> {
        self.requested_devices
    }
    #[cfg(test)]
    pub(crate) const fn policy(self) -> PlacementPolicy {
        self.policy
    }
    /// Index is a layout device-list slot, not a logical rank. In root/expert
    /// layout the local root and expert zero both use slot zero.
    pub(crate) fn device(self, slot: usize) -> Option<usize> {
        if self.target != PlacementTarget::Device || slot >= self.topology.device_slots {
            return None;
        }
        Some(self.requested_devices.map_or(slot, |devices| devices[slot]))
    }
}

pub(crate) fn validate_placement<'a>(
    topology: ValidatedTopology,
    target: PlacementTarget,
    devices: Option<&'a [usize]>,
    policy: PlacementPolicy,
) -> Result<ValidatedPlacement<'a>> {
    match target {
        PlacementTarget::Host => {
            if devices.is_some() {
                return Err(PolicyError::DevicePlacementOnHost);
            }
            if policy.selection == DeviceSelection::Explicit {
                return Err(PolicyError::ExplicitDevicesRequired);
            }
        }
        PlacementTarget::Device => match devices {
            Some(devices) => {
                if devices.len() != topology.device_slots {
                    return Err(PolicyError::DeviceCount {
                        expected: topology.device_slots,
                        actual: devices.len(),
                    });
                }
                if devices.iter().any(|&d| i32::try_from(d).is_err()) {
                    return Err(PolicyError::DeviceOrdinal);
                }
                if policy.sharing == DeviceSharing::Distinct
                    && devices
                        .iter()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        != devices.len()
                {
                    return Err(PolicyError::DuplicateDevice);
                }
            }
            None => {
                if policy.selection == DeviceSelection::Explicit {
                    return Err(PolicyError::ExplicitDevicesRequired);
                }
                // Preserve the existing conservative default-placement ABI bound.
                if topology.device_slots > i32::MAX as usize {
                    return Err(PolicyError::DeviceOrdinal);
                }
            }
        },
    }
    Ok(ValidatedPlacement {
        target,
        topology,
        policy,
        requested_devices: devices,
    })
}

pub(crate) fn validate_defaults(checks: &[(&'static str, bool)]) -> Result<()> {
    for &(name, is_default) in checks {
        validate_override(name, Some(&is_default), &true, OverridePolicy::DefaultOnly)?;
    }
    Ok(())
}
