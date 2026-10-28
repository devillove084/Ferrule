use super::experts::OwnerPlan;
use super::inputs::ExpertPlacementConfig;
use super::*;
use ferrule_model::transformer::FeedForward;
pub(in crate::model_factory) const OWNER_WORKSPACE: usize = super::EXPERT_WORKSPACE;
use ferrule_model::transformer::BoundDecoderResources;
use ferrule_model::transformer::host_experts::{
    HostCacheMemoryPlan, HostExpertCacheOptions, HostExpertWarmPlan, HostMemoryBudget,
    HostMemorySnapshot, StandardHostStartupMemory,
};
use ferrule_runtime::parallel::expert::ExpertParallelExecutor;
use ferrule_runtime::parallel::expert_process::{FrameSizeOverflow, NumericFrameSize};
use ferrule_runtime::parallel::expert_process::{
    NumericBoot, NumericProcessIsolation, owned_catalog,
};
use ferrule_runtime::parallel::process::decoder::{DecoderDevice, ExpertPlacementFrame};
use ferrule_runtime::parallel::process::{
    ProcessGroupEpoch, ProcessIdentity, ProcessLaunch, ProcessOwnerConfig, ProcessOwnerInstanceId,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

pub(in crate::model_factory) struct Plan {
    pub(in crate::model_factory) launch: ProcessLaunch,
    pub(in crate::model_factory) boots: Vec<(ProcessIdentity, NumericBoot)>,
    pub(in crate::model_factory) options: ProcessOwnerConfig,
    pub(in crate::model_factory) cache: HostCacheMemoryPlan,
    pub(in crate::model_factory) budget: HostMemoryBudget,
}

impl Plan {
    pub(in crate::model_factory) fn new(
        request: &ResolvedModelRequest,
        resources: &BoundDecoderResources,
        placement: &ExpertPlacementConfig,
        owners: &OwnerPlan,
        host_options: HostExpertCacheOptions,
        profile: NumericFp8Profile,
    ) -> Result<Self> {
        let launch = placement
            .process_launch()
            .ok_or_else(|| provider_unsupported("numeric process EP requires process launch"))?;
        if owners.group.placement.epoch().get() != 0 || owners.group.placement.is_replicated() {
            return Err(provider_unsupported(
                "numeric process supports only fixed single-replica placement (epoch 0); versioned or replicated routes require an unsupported residency/provider authority",
            ));
        }
        let options = fit_frames(
            placement.process_owner_config()?,
            resources.spec().hidden_size(),
            owners.group.limits.max_tokens,
        )?;
        let total_layers = resources.spec().layers().len();
        let base = StandardHostStartupMemory::for_resources(
            resources,
            request.driver_config.ctx_size,
            request.max_tensor_bytes,
        )?;
        let child_host_options = HostExpertCacheOptions {
            memory_budget: HostMemoryBudget {
                headroom_bytes: host_options.memory_budget.headroom_bytes,
                ..Default::default()
            },
            ..host_options
        };
        let mut boots = Vec::with_capacity(owners.group.members.len());
        let mut partition_plans = Vec::with_capacity(owners.group.members.len());
        static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);
        let epoch = ProcessGroupEpoch::new(
            NEXT_EPOCH
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| provider_unsupported("numeric EP process epoch exhausted"))?,
        )
        .map_err(|e| provider_unsupported(e.to_string()))?;
        if resources.host_experts().is_some() {
            return Err(provider_unsupported(
                "process root must not retain a full routed host image",
            ));
        }
        let mut entries = Vec::new();
        for layer in &resources.spec().layers()[owners.group.layers.clone()] {
            let FeedForward::Moe(moe) = layer.feed_forward() else {
                unreachable!()
            };
            for expert in 0..moe.router_spec().num_experts() {
                entries.push((
                    layer.index(),
                    expert,
                    owners
                        .group
                        .placement
                        .owner(layer.index(), expert)
                        .ok_or_else(|| provider_unsupported("missing scoped expert placement"))?
                        .get(),
                ));
            }
        }
        let frame = ExpertPlacementFrame {
            source_scope: ferrule_common::topology::ExpertSourceScope::ExternalStage,
            source: owners.group.source_rank.get(),
            members: owners.group.members.iter().map(|rank| rank.get()).collect(),
            entries,
            route_entries: None,
            placement_epoch: 0,
            max_tokens: owners.group.limits.max_tokens,
            max_bytes: owners.group.limits.max_bytes,
            devices: placement
                .expert_devices()
                .iter()
                .map(|&ordinal| DecoderDevice::Cuda { ordinal })
                .collect(),
            timeout_ms: u64::try_from(placement.rank_timeout().as_millis())
                .map_err(|_| provider_unsupported("numeric EP timeout exceeds wire range"))?,
        };
        for (slot, owner) in owners.group.members.iter().copied().enumerate() {
            let expected = owned_catalog(resources, &owners.group, owner)?;
            let owned = expected.iter().map(|source| (source.layer, source.expert));
            let warm = HostExpertWarmPlan::for_owned(
                resources.state_dict(),
                child_host_options,
                request.max_tensor_bytes,
                owned,
            )?;
            if warm.payload_bytes() != owners.weights[slot] as u64
                || warm.experts() != owners.counts[slot]
            {
                return Err(provider_unsupported(
                    "owned catalog/warm payload differs from physical plan",
                ));
            }
            partition_plans.push(warm.memory_plan());
            let boot = NumericBoot {
                numeric_expert_boot:
                    ferrule_runtime::parallel::expert_process::NUMERIC_EXPERT_WIRE_VERSION,
                numeric_profile: profile.precision(),
                checkpoint: request.model_path.clone(),
                total_layers,
                layers: owners.group.layers.clone(),
                placement: frame.clone(),
                owner: owner.get(),
                hidden: resources.spec().hidden_size(),
                max_rows: request.scheduler_config.max_batch_tokens,
                max_parameter_bytes: request.max_tensor_bytes,
                host_max_bytes: warm.required_bytes(),
                host_max_experts: warm.experts(),
                host_workers: host_options.workers,
                host_headroom_bytes: host_options.memory_budget.headroom_bytes,
                device_bytes: super::budget::expert_payload_envelope(
                    owners.weights[slot],
                    owners.activation_bytes,
                )?,
                workspace_bytes: OWNER_WORKSPACE,
                expected,
            };
            boot.group()?;
            boot.validate_frame_limits(options.frame_limits)?;
            let bytes =
                serde_json::to_vec(&boot).map_err(|e| provider_unsupported(e.to_string()))?;
            if bytes.len() > options.frame_limits.max_config_bytes
                || bytes
                    .len()
                    .checked_add(1024)
                    .is_none_or(|n| n > options.frame_limits.max_frame_bytes)
            {
                return Err(provider_unsupported(
                    "numeric child boot exceeds config frame",
                ));
            }
            let instance = ProcessOwnerInstanceId::new(slot as u64 + 1)
                .map_err(|e| provider_unsupported(e.to_string()))?;
            boots.push((ProcessIdentity::new(epoch, owner, instance), boot));
        }
        let mut cache = HostCacheMemoryPlan::serial_partitions(&partition_plans)?;
        if cache.peak_bytes()? > host_options.max_bytes
            || owners
                .counts
                .iter()
                .try_fold(0usize, |sum, n| sum.checked_add(*n))
                .is_none_or(|count| count > host_options.max_experts)
        {
            return Err(provider_unsupported(
                "aggregate owned host partitions exceed global cache cap/count",
            ));
        }
        // Each child rebinds the complete metadata, but no unowned payload. The
        // conservative per-binding ledger covers catalogs/maps/paths; charge it
        // per process, plus bounded JSON DOM/frame copies and owner stacks.
        let extra = super::budget::process_child_overhead(
            base.metadata_bytes,
            options.frame_limits.max_frame_bytes as u64,
            boots.len() as u64,
        )?;
        cache.retained_metadata_bytes = cache
            .retained_metadata_bytes
            .checked_add(extra)
            .ok_or_else(|| provider_unsupported("process retained metadata overflow"))?;
        if cache.peak_bytes()? > host_options.max_bytes {
            return Err(provider_unsupported(
                "aggregate host peak including IPC metadata exceeds global cap",
            ));
        }
        let mut budget = base.memory_budget()?;
        budget.headroom_bytes = host_options.memory_budget.headroom_bytes;
        // No RSS credit: root payload is built only after every child is Ready.
        Ok(Self {
            launch,
            boots,
            options,
            cache,
            budget,
        })
    }

    pub(in crate::model_factory) fn admit_live(&self) -> Result<()> {
        self.admit(memory::snapshot()?)
    }

    pub(in crate::model_factory) fn admit(&self, snapshot: HostMemorySnapshot) -> Result<()> {
        self.budget.admit(self.cache, snapshot)?;
        Ok(())
    }

    pub(in crate::model_factory) fn start(self) -> Result<ExpertParallelExecutor> {
        self.admit_live()?;
        Ok(ExpertParallelExecutor::new_numeric_process_scoped(
            self.launch,
            self.boots,
            self.options,
            NumericProcessIsolation::Standalone,
        )?)
    }
}

#[cfg(any(feature = "cuda", test))]
pub(in crate::model_factory) use crate::model_factory::parallel::experts::check_process_ready as check_ready;

pub(in crate::model_factory) fn fit_frames(
    mut options: ProcessOwnerConfig,
    hidden: usize,
    max_tokens: usize,
) -> Result<ProcessOwnerConfig> {
    let size = NumericFrameSize::new(hidden, max_tokens).map_err(|overflow| match overflow {
        FrameSizeOverflow::Payload => provider_unsupported("numeric process frame overflow"),
        FrameSizeOverflow::Envelope => {
            provider_unsupported("numeric routed batch exceeds 8 MiB process frame")
        }
    })?;
    if size.frame_bytes() > options.frame_limits.max_frame_bytes {
        return Err(provider_unsupported(
            "numeric routed batch exceeds 8 MiB process frame",
        ));
    }
    // Preserve generic child ceilings for small batches. Large numeric buckets
    // require a trusted endpoint accepting the full existing 8 MiB ceiling.
    options.frame_limits.max_command_bytes = options
        .frame_limits
        .max_command_bytes
        .max(size.payload_bytes());
    options.frame_limits.max_output_bytes = options
        .frame_limits
        .max_output_bytes
        .max(size.payload_bytes());
    Ok(options)
}

impl ExpertPlacementConfig {
    pub(in crate::model_factory) fn rank_timeout(&self) -> Duration {
        self.rank_timeout
    }

    #[cfg(unix)]
    pub(in crate::model_factory) fn process_launch(&self) -> Option<ProcessLaunch> {
        self.process_launch.clone()
    }

    #[cfg(unix)]
    pub(in crate::model_factory) fn process_owner_config(&self) -> Result<ProcessOwnerConfig> {
        let config =
            crate::model_factory::parallel::inputs::process_owner_config(self.rank_timeout);
        config
            .validate()
            .map_err(|e| provider_unsupported(e.to_string()))?;
        Ok(config)
    }
}

use crate::model_factory::host_memory as memory;
