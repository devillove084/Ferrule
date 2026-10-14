//! Scoped ownership and immutable Ready contracts shared by resident and pipeline roots.
use super::inputs::ExpertPlacementConfig;
use super::*;
#[cfg(feature = "cuda")]
use ferrule_model::decoder::HybridCudaRoutedExecutor;
use ferrule_model::nn::ParameterResidency;
use ferrule_model::transformer::{
    FeedForward,
    expert_parallel::{ExpertDispatchLimits, ExpertPlacement},
};
use ferrule_runtime::parallel::expert::ExpertGroup;
#[cfg(feature = "cuda")]
use ferrule_runtime::parallel::expert::{
    ExpertParallelExecutor, ImmutableOwnedAttachment, ImmutableOwnedAttachmentConfig,
    ImmutableOwnedBaseline, NumericExpertConfig,
};
#[cfg(feature = "cuda")]
use std::{rc::Rc, sync::Arc};
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| provider_unsupported("EP memory estimate overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| provider_unsupported("EP memory estimate overflow"))
}

#[derive(Clone)]
pub(in crate::model_factory) struct OwnerPlan {
    pub(in crate::model_factory) group: ExpertGroup,
    pub(in crate::model_factory) weights: Vec<usize>,
    pub(in crate::model_factory) counts: Vec<usize>,
    pub(in crate::model_factory) activation_bytes: usize,
}

impl OwnerPlan {
    pub(in crate::model_factory) fn new(
        resources: &BoundDecoderResources,
        placement: &ExpertPlacementConfig,
        rows: usize,
    ) -> Result<Self> {
        Self::for_stage(
            resources,
            placement,
            rows,
            0..resources.spec().layers().len(),
            ParallelRankId::new(0),
            (1..=placement.expert_parallel())
                .map(|n| ParallelRankId::new(n as u32))
                .collect(),
        )
    }

    pub(in crate::model_factory) fn for_stage(
        resources: &BoundDecoderResources,
        placement: &ExpertPlacementConfig,
        rows: usize,
        layers: std::ops::Range<usize>,
        source_rank: ParallelRankId,
        members: Vec<ParallelRankId>,
    ) -> Result<Self> {
        Self::for_stage_degree(
            resources,
            placement.expert_parallel(),
            rows,
            layers,
            source_rank,
            members,
        )
    }

    pub(in crate::model_factory) fn for_stage_degree(
        resources: &BoundDecoderResources,
        degree: usize,
        rows: usize,
        layers: std::ops::Range<usize>,
        source_rank: ParallelRankId,
        members: Vec<ParallelRankId>,
    ) -> Result<Self> {
        if layers.is_empty()
            || layers.end > resources.spec().layers().len()
            || rows == 0
            || members.len() != degree
            || members.contains(&source_rank)
            || members
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != members.len()
        {
            return Err(provider_unsupported(
                "invalid scoped EP range/source/membership",
            ));
        }
        let mut entries = Vec::new();
        let mut counts = vec![0; members.len()];
        let mut max_tokens = 0;
        for layer in &resources.spec().layers()[layers.clone()] {
            let FeedForward::Moe(moe) = layer.feed_forward() else {
                return Err(provider_unsupported(
                    "numeric EP requires routed experts in every layer",
                ));
            };
            let experts = moe.router_spec().num_experts();
            if degree < 2 || !experts.is_multiple_of(degree) {
                return Err(provider_unsupported(
                    "stage expert count must be divisible by the external owner count",
                ));
            }
            max_tokens = max_tokens.max(mul(rows, moe.router_spec().experts_per_token())?);
            for expert in 0..experts {
                let slot = expert % members.len();
                counts[slot] += 1;
                entries.push((layer.index(), expert, members[slot]));
            }
        }
        let mut activation_bytes = 0;
        for layer in &resources.spec().layers()[layers.clone()] {
            let FeedForward::Moe(moe) = layer.feed_forward() else {
                unreachable!()
            };
            activation_bytes = activation_bytes.max(
                moe.expert()
                    .f32_scratch_bytes(max_tokens)
                    .map_err(|_| provider_unsupported("EP memory estimate overflow"))?,
            );
        }
        let group = ExpertGroup {
            source_rank,
            members,
            layers: layers.clone(),
            placement: ExpertPlacement::new(entries)?,
            limits: ExpertDispatchLimits {
                max_tokens,
                max_bytes: mul(mul(max_tokens, resources.spec().hidden_size())?, 4)?,
            },
        };
        let mut weights = vec![0; degree];
        for parameter in resources.state_dict().parameters() {
            if let ParameterResidency::Expert { layer, expert } = parameter.residency() {
                if !layers.contains(layer) {
                    continue;
                }
                let owner = group
                    .placement
                    .owner(*layer, *expert)
                    .ok_or_else(|| provider_unsupported("unplaced routed parameter"))?;
                let slot = group
                    .members
                    .iter()
                    .position(|member| *member == owner)
                    .ok_or_else(|| provider_unsupported("unmapped expert owner"))?;
                let bytes = parameter
                    .weight()
                    .slice()
                    .bytes
                    .checked_add(parameter.scale().map_or(0, |scale| scale.slice().bytes))
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| provider_unsupported("expert payload bytes overflow"))?;
                weights[slot] = add(weights[slot], bytes)?;
            }
        }
        Ok(Self {
            group,
            weights,
            counts,
            activation_bytes,
        })
    }
}

#[cfg(feature = "cuda")]
pub(in crate::model_factory) fn check_ready(
    plan: &OwnerPlan,
    stats: &[ferrule_runtime::parallel::expert::ExpertOwnerStats],
) -> ferrule_common::Result<()> {
    let invalid = || ferrule_common::Error::Execution {
        message: "actual expert Ready differs from sealed placement/residency".into(),
    };
    if stats.len() != plan.group.members.len() {
        return Err(invalid());
    }
    for (slot, stats) in stats.iter().enumerate() {
        let cache = stats.cuda_cache.ok_or_else(invalid)?;
        let expected = plan
            .group
            .layers
            .clone()
            .flat_map(|layer| {
                (0..plan.counts.iter().sum::<usize>() / plan.group.layers.len())
                    .filter(move |&expert| {
                        plan.group.placement.owner(layer, expert) == Some(plan.group.members[slot])
                    })
                    .map(move |expert| (layer, expert))
            })
            .collect::<std::collections::BTreeSet<_>>();
        let actual = stats
            .owned_experts
            .iter()
            .map(|e| (e.layer, e.expert))
            .collect::<std::collections::BTreeSet<_>>();
        if stats.rank.owner != plan.group.members[slot]
            || stats.rank.slot.get() as usize != slot
            || actual != expected
            || stats.owned_experts.len() != plan.counts[slot]
            || cache.resident_experts != plan.counts[slot]
            || cache.resident_bytes != plan.weights[slot]
            || cache.uploads != plan.counts[slot] as u64
            || cache.evictions != 0
            || cache.pending_upload_bytes != 0
            || cache.quarantined
        {
            return Err(invalid());
        }
    }
    Ok(())
}

/// Start eager numeric owners against the exact shared checkpoint/cache authority.
#[cfg(feature = "cuda")]
pub(in crate::model_factory) fn start_threads(
    plan: &OwnerPlan,
    resources: Arc<BoundDecoderResources>,
    devices: Vec<usize>,
    config: NumericExpertConfig,
    profile: NumericFp8Profile,
) -> ferrule_common::Result<ExpertParallelExecutor> {
    let host =
        resources
            .host_experts()
            .cloned()
            .ok_or_else(|| ferrule_common::Error::Execution {
                message: "expert startup requires the admitted shared host image".into(),
            })?;
    ExpertParallelExecutor::new_numeric_with_profile(
        plan.group.clone(),
        resources,
        host,
        devices,
        config,
        profile,
    )
}

/// Emit the primary error before cleanup can panic. Runtime recognizes the
/// exact PanicQuiescence payload; do not wrap it or catch it as an ordinary Err.
/// The event retains structured error/source information even when no combined
/// return value is possible. Resource custody remains with runtime shutdown.
pub(in crate::model_factory) fn report_admission_failure(
    operation: &'static str,
    source: &(dyn std::error::Error + 'static),
) {
    tracing::error!(
        target: "ferrule_composition::parallel::expert_admission",
        operation,
        phase = "ready",
        cleanup = "not_started",
        error = source,
        "expert owner admission failed before cleanup",
    );
}

fn finish_ready<T>(
    admission: ferrule_common::Result<T>,
    cleanup: impl FnOnce() -> ferrule_common::Result<()>,
) -> ferrule_common::Result<T> {
    finish_admission(
        "expert Ready",
        admission,
        cleanup,
        ferrule_common::Error::with_cleanup,
    )
}

/// Capture immutable Ready evidence. The caller still owns the executor and
/// must preserve its shutdown evidence if this or subsequent admission fails.
#[cfg(feature = "cuda")]
pub(in crate::model_factory) fn thread_baseline(
    experts: &mut ExpertParallelExecutor,
    plan: &OwnerPlan,
    devices: Vec<usize>,
    host: Option<Arc<ferrule_model::transformer::host_experts::HostExpertCache>>,
) -> ferrule_common::Result<ImmutableOwnedBaseline> {
    let stats = experts.owner_stats()?;
    check_ready(plan, &stats)?;
    let hits = host.as_ref().map_or(0, |h| h.hits());
    ImmutableOwnedBaseline::threads(stats, devices, host, hits)
}

#[cfg(feature = "cuda")]
pub(super) fn prepare_attachment(
    plan: &OwnerPlan,
    resources: Arc<BoundDecoderResources>,
    devices: Vec<usize>,
    config: PipelineConfig,
    profile: NumericFp8Profile,
    timeout: std::time::Duration,
    device: &ferrule_model::decoder::HybridCudaDevice,
) -> ferrule_common::Result<Box<dyn HybridCudaRoutedExecutor>> {
    let attachment_config = ImmutableOwnedAttachmentConfig::new(
        plan.group.source_rank,
        plan.group.layers.end - 1,
        timeout,
    )?;
    let limit = super::budget::expert_payload_envelope(
        *plan.weights.iter().max().unwrap(),
        plan.activation_bytes,
    )
    .map_err(|e| ferrule_common::Error::Execution {
        message: e.to_string(),
    })?;
    let mut experts = start_threads(
        plan,
        resources,
        devices.clone(),
        NumericExpertConfig {
            max_parameter_bytes: config.max_parameter_bytes,
            max_device_bytes: limit,
            workspace_bytes: EXPERT_WORKSPACE,
            dispatch_timeout: timeout,
        },
        profile,
    )?;
    // Pipeline stages share the image, so global host hits are not stage-local.
    let ready = thread_baseline(&mut experts, plan, devices, None);
    let baseline = finish_ready(ready, || experts.shutdown())?;
    let adapter = experts.into_hybrid_cuda(Rc::clone(device.operators()))?;
    Ok(Box::new(ImmutableOwnedAttachment::new(
        adapter,
        attachment_config,
        baseline,
    )))
}

#[cfg(unix)]
pub(in crate::model_factory) fn check_process_ready(
    owners: &OwnerPlan,
    placement: &ExpertPlacementConfig,
    stats: &[ferrule_runtime::parallel::expert_process::NumericProcessStats],
) -> Result<()> {
    if stats.len() != owners.group.members.len()
        || stats
            .iter()
            .map(|s| s.pid)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != stats.len()
        || stats.iter().enumerate().any(|(slot, s)| {
            s.pid == 0
                || s.pid == std::process::id()
                || s.owner != owners.group.members[slot].get()
                || s.slot as usize != slot
                || s.device
                    != (ferrule_runtime::parallel::process::decoder::DecoderDevice::Cuda {
                        ordinal: placement.expert_devices()[slot],
                    })
                || s.owned_experts.len() != owners.counts[slot]
                || s.resident_experts != owners.counts[slot]
                || s.resident_bytes != owners.weights[slot]
                || s.host_payload_bytes != owners.weights[slot] as u64
                || s.host_read_bytes != owners.weights[slot] as u64
                || s.uploads != owners.counts[slot] as u64
                || s.evictions != 0
                || s.pending_upload_bytes != 0
                || s.quarantined
        })
    {
        return Err(provider_unsupported(
            "process Ready differs from admitted owned physical plan",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/model_factory/parallel_expert_cleanup.rs"]
mod cleanup_tests;

pub(in crate::model_factory) fn finish_owner_admission<T>(
    operation: &'static str,
    admission: Result<T>,
    cleanup: impl FnOnce() -> Result<()>,
) -> Result<T> {
    finish_admission(
        operation,
        admission,
        cleanup,
        |operation, source, cleanup| match cleanup {
            Ok(()) => source,
            Err(cleanup) => Error::Cleanup {
                operation,
                source: Box::new(source),
                cleanup: Box::new(cleanup),
            },
        },
    )
}

fn finish_admission<T, E: std::error::Error + 'static>(
    operation: &'static str,
    admission: std::result::Result<T, E>,
    cleanup: impl FnOnce() -> std::result::Result<(), E>,
    combine: impl FnOnce(&'static str, E, std::result::Result<(), E>) -> E,
) -> std::result::Result<T, E> {
    admission.map_err(|source| {
        report_admission_failure(operation, &source);
        combine(operation, source, cleanup())
    })
}
