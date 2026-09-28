//! Qwen3.5 resident-root composition with eager, GPU-thread expert owners.
use super::*;
use crate::parallel::expert::{
    ExpertGroup, ExpertOwnerStats, ExpertParallelExecutor, HybridCudaExpertAdapter,
    NumericExpertConfig,
};
use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_common::ParallelRankId;
use ferrule_model::decoder::{
    HybridCudaDecoder, HybridCudaDevice, HybridCudaExpertProgress, HybridCudaMemoryBudget,
    HybridCudaRoutedExecutor, HybridCudaRoutedExperts,
};
use ferrule_model::nn::ParameterResidency;
use ferrule_model::transformer::expert_parallel::{
    ExpertDispatchLimits, ExpertPlacement, RoutedSwiGluExecutor, RoutedSwiGluRequest,
};
use ferrule_model::transformer::{
    ExpertCacheLimits, FeedForward, NumericFp8Precision, OperatorProgress, Rows,
};
use std::{
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};

#[cfg(test)]
#[path = "qwen35_ep_strict_tests.rs"]
mod strict;
#[cfg(test)]
#[path = "qwen35_ep_tests.rs"]
mod tests;

const OWNER_WORKSPACE: usize = 64 << 20;
const DISPATCH_TIMEOUT: Duration = Duration::from_secs(30);

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| ep_unsupported("EP memory estimate overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| ep_unsupported("EP memory estimate overflow"))
}

struct OwnerPlan {
    group: ExpertGroup,
    weights: Vec<usize>,
    counts: Vec<usize>,
    activation_bytes: usize,
}

impl OwnerPlan {
    fn new(
        resources: &BoundDecoderResources,
        placement: &Qwen35ExpertPlacement,
        rows: usize,
    ) -> Result<Self> {
        let members = (1..=placement.expert_parallel())
            .map(|n| ParallelRankId::new(n as u32))
            .collect::<Vec<_>>();
        let source_rank = ParallelRankId::new(0);
        let mut entries = Vec::new();
        let mut counts = vec![0; members.len()];
        let mut max_tokens = 0;
        for layer in resources.spec().layers() {
            let FeedForward::Moe(moe) = layer.feed_forward() else {
                return Err(ep_unsupported(
                    "Qwen3.5 EP requires routed experts in every layer",
                ));
            };
            let experts = moe.router_spec().num_experts();
            placement.validate_expert_shape(1, experts)?;
            max_tokens = max_tokens.max(mul(rows, moe.router_spec().experts_per_token())?);
            for expert in 0..experts {
                let slot = expert % members.len();
                counts[slot] += 1;
                entries.push((layer.index(), expert, members[slot]));
            }
        }
        let mut activation_bytes = 0;
        for layer in resources.spec().layers() {
            let FeedForward::Moe(moe) = layer.feed_forward() else {
                unreachable!()
            };
            let width = add(
                mul(moe.expert().gate().out_features(), 3)?,
                mul(resources.spec().hidden_size(), 2)?,
            )?;
            activation_bytes = activation_bytes.max(mul(mul(width, max_tokens)?, 4)?);
        }
        let group = ExpertGroup {
            source_rank,
            members,
            layers: 0..resources.spec().layers().len(),
            placement: ExpertPlacement::new(entries)?,
            limits: ExpertDispatchLimits {
                max_tokens,
                max_bytes: mul(mul(max_tokens, resources.spec().hidden_size())?, 4)?,
            },
        };
        let mut weights = vec![0; placement.expert_parallel()];
        for parameter in resources.state_dict().parameters() {
            if let ParameterResidency::Expert { layer, expert } = parameter.residency() {
                let owner = group
                    .placement
                    .owner(*layer, *expert)
                    .ok_or_else(|| ep_unsupported("unplaced routed parameter"))?;
                let slot = owner.get() as usize - 1;
                let bytes = parameter
                    .weight()
                    .slice()
                    .bytes
                    .checked_add(parameter.scale().map_or(0, |scale| scale.slice().bytes))
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| ep_unsupported("expert payload bytes overflow"))?;
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

/// Make the model attachment's single shutdown poll a bounded join boundary,
/// including failed runner preparation. Never report Complete after timeout.
struct JoinedExperts {
    adapter: HybridCudaExpertAdapter,
    baseline: Vec<ExpertOwnerStats>,
    devices: Vec<usize>,
    source: ParallelRankId,
    last_layer: usize,
    host: Arc<ferrule_model::transformer::host_experts::HostExpertCache>,
    ready_host_hits: u64,
}
impl RoutedSwiGluExecutor for JoinedExperts {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        active: &mut dyn FnMut(
            ferrule_common::execution::ExecutionTransactionId,
        ) -> ferrule_common::Result<()>,
    ) -> ferrule_common::Result<OperatorProgress<Rows>> {
        let context = request.context;
        if context.source_rank != self.source {
            return Err(ferrule_common::Error::Execution {
                message: "EP source identity changed".into(),
            });
        }
        let result = self.adapter.routed_swiglu(request, active)?;
        if context.layer == self.last_layer && matches!(result, OperatorProgress::Ready(_)) {
            let stats = self.adapter.owner_stats()?;
            if stats.len() != self.baseline.len() {
                return Err(ferrule_common::Error::Execution {
                    message: "EP owner count changed after forward".into(),
                });
            }
            for (slot, (before, after)) in self.baseline.iter().zip(&stats).enumerate() {
                let cache = after
                    .cuda_cache
                    .ok_or_else(|| ferrule_common::Error::Execution {
                        message: "EP CUDA stats missing".into(),
                    })?;
                let initial = before.cuda_cache.expect("validated CUDA Ready stats");
                if before.rank != after.rank
                    || before.owned_experts != after.owned_experts
                    || cache.uploads != initial.uploads
                    || cache.evictions != 0
                    || cache.resident_bytes != initial.resident_bytes
                    || cache.resident_experts != initial.resident_experts
                    || cache.pending_upload_bytes != 0
                    || cache.quarantined
                {
                    return Err(ferrule_common::Error::Execution {
                        message: "EP immutable owned residency changed during forward".into(),
                    });
                }
                tracing::info!(
                    transaction = context.transaction.get(),
                    source = self.source.get(),
                    layer = context.layer,
                    owner = after.rank.owner.get(),
                    ordinal = self.devices[slot],
                    calls = after.calls,
                    tokens = after.tokens,
                    uploads_before = initial.uploads,
                    uploads_after = cache.uploads,
                    evictions = cache.evictions,
                    resident_bytes = cache.resident_bytes,
                    resident_experts = cache.resident_experts,
                    "Qwen3.5 EP actual drained owner stats"
                );
            }
            if self.host.hits() != self.ready_host_hits {
                return Err(ferrule_common::Error::Execution {
                    message: "EP unexpectedly rematerialized host experts after Ready".into(),
                });
            }
            tracing::info!(generation=self.host.generation(), identity=?Arc::as_ptr(&self.host), hits=self.host.hits(),
                "Qwen3.5 EP unchanged shared host image after forward");
        }
        Ok(result)
    }
}
impl HybridCudaRoutedExecutor for JoinedExperts {
    fn drain(&mut self) -> ferrule_common::Result<HybridCudaExpertProgress> {
        self.adapter.drain()
    }
    fn on_error(&mut self, transaction: ferrule_common::execution::ExecutionTransactionId) {
        self.adapter.on_error(transaction);
    }
    fn shutdown(&mut self) -> ferrule_common::Result<HybridCudaExpertProgress> {
        let deadline = Instant::now() + DISPATCH_TIMEOUT;
        loop {
            match self.adapter.shutdown()? {
                HybridCudaExpertProgress::Pending if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_micros(50))
                }
                HybridCudaExpertProgress::Pending => return Ok(HybridCudaExpertProgress::Unknown),
                progress => return Ok(progress),
            }
        }
    }
}

fn check_physical(plan: &Qwen35ExpertPhysicalPlan, device: &HybridCudaDevice) -> Result<()> {
    let mut capacities = Vec::with_capacity(plan.cards().len());
    for card in plan.cards() {
        let (free_bytes, total_bytes) = if card.ordinal == device.device_ordinal() {
            device.operators().memory_info()?
        } else {
            // A temporary probe has no model weights; drop it before uploads.
            CudaOperators::new_on_device(card.ordinal)?.memory_info()?
        };
        capacities.push(Qwen35PhysicalCardCapacity {
            ordinal: card.ordinal,
            free_bytes,
            limit_bytes: total_bytes,
        });
    }
    plan.check_capacity(&capacities)
}

pub(super) fn build(request: ResolvedModelRequest) -> Result<BoxedSessionInferenceEngine> {
    if request.family != ModelFamily::Qwen35Moe
        || request.backend != ModelExecutionBackend::Cuda
        || request.pipeline.is_some()
    {
        return Err(ep_unsupported(
            "Qwen3.5 EP requires a CUDA resident root, not CPU or Pipeline",
        ));
    }
    let (_, resources) = Qwen35Adapter::bind_hf_metadata(&request.model_path)?;
    let tokenizer = TokenizerHandle::load(&request.model_path)?;
    build_bound(request, resources, tokenizer)
}

struct PreparedRoot {
    runner: GenericDecoderRunner<HybridCudaDecoder>,
    planes: ferrule_model::decoder::StandardGqaPlanes,
    accounting: ResidentKvPageAccounting,
    host: Arc<ferrule_model::transformer::host_experts::HostExpertCache>,
    limits: Qwen35MoeCapacityLimits,
    #[cfg(test)]
    device: HybridCudaDevice,
}

fn build_bound(
    request: ResolvedModelRequest,
    resources: BoundDecoderResources,
    tokenizer: TokenizerHandle,
) -> Result<BoxedSessionInferenceEngine> {
    let prepared = prepare_bound(&request, resources, tokenizer)?;
    let engine = super::super::super::composition::compose_resident_engine(
        prepared.runner,
        Box::new(prepared.planes),
        prepared.accounting,
        request.scheduler_config,
        request.driver_config,
    )?;
    Ok(Box::new(ReportedMoeEngine {
        engine,
        host_experts: Some(prepared.host),
        limits: prepared.limits,
        expert_placement: request
            .family_options
            .qwen35()
            .and_then(|options| options.expert_placement())
            .cloned(),
        shutdown_attempted: false,
        close_reported: false,
    }))
}

// Shared constructor boundary for the resident composition and opt-in strict
// full-logit tests. No second weight image or alternate forward is created.
fn prepare_bound(
    request: &ResolvedModelRequest,
    resources: BoundDecoderResources,
    tokenizer: TokenizerHandle,
) -> Result<PreparedRoot> {
    let family = request
        .family_options
        .qwen35()
        .expect("validated Qwen35Moe");
    let moe = family.moe().expect("validated Qwen35Moe");
    let placement = family
        .expert_placement()
        .ok_or_else(|| ep_unsupported("missing explicit EP placement"))?;
    if request.max_layers != resources.spec().layers().len()
        || moe.device_capacity.is_some()
        || request.expert_memory_policy != ExpertMemoryPolicy::default()
        || request.moe_hotset_experts != 0
        || request.expert_reader_max_tensor_bytes != 64 << 20
        || request.output_head_chunk_rows != 4096
        || request.driver_config.enable_native_proposals
        || request.scheduler_config.prefix_cache_capacity_pages != 0
    {
        return Err(ep_unsupported(
            "Qwen3.5 EP requires full depth and does not consume local cache, pinned cache, hotset, reader/head overrides, speculation or prefix cache",
        ));
    }
    let limits = capacity_limits(&request)?;
    resources.validate_parameter_limits(request.max_tensor_bytes, request.max_tensor_bytes)?;
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Qwen35Moe,
        WeightSource::Safetensors,
        16,
        limits.max_positions,
        limits.max_batch_tokens,
        limits.max_sequences,
        request.max_tensor_bytes,
        ExecutionPrecisionPolicy::f32(),
    )?
    .with_hybrid_cuda_numeric_fp8_precision(
        ExpertCacheLimits {
            max_experts: limits.max_experts,
            max_bytes: limits.max_bytes,
        },
        limits.scratch_bytes,
        NumericFp8Precision::F32Tf32x3,
    )?;
    let planes = HybridStateSchema::from_spec(resources.spec(), request.max_layers)?
        .kv_planes(options.page_size(), options.max_positions())?;
    let accounting = ResidentKvPageAccounting::TransactionCapacity {
        page_bytes: physical_page_bytes(&planes)?,
        budget_bytes: Some(limits.kv_bytes as u64),
    };
    let kv = plan_resident_kv_pages(
        &planes,
        accounting,
        request.scheduler_config,
        request.driver_config,
    )?;
    let root = Qwen35ExpertRootMemory::for_resources(&resources, &options, kv.configured_pages)?;
    let owners = OwnerPlan::new(&resources, placement, limits.max_batch_tokens)?;
    let owner_workspace = add(OWNER_WORKSPACE, owners.activation_bytes)?;
    let physical = Qwen35ExpertPhysicalPlan::new(
        placement,
        root,
        &owners.weights,
        owner_workspace,
        limits.allocator_margin_bytes,
    )?;
    let owner_limit = add(
        add(
            *owners
                .weights
                .iter()
                .max()
                .ok_or_else(|| ep_unsupported("empty EP ownership"))?,
            OWNER_WORKSPACE,
        )?,
        owners.activation_bytes,
    )?;
    let device = HybridCudaDevice::new_on_device(
        placement.root_device(),
        HybridCudaMemoryBudget {
            kv_pages: kv.configured_pages,
            state_bytes: root.state_bytes,
            weight_bytes: root.weights_bytes,
            workspace_bytes: root.workspace_bytes,
        },
    )?;
    check_physical(&physical, &device)?;
    let host_options = host_startup_options(
        &resources,
        moe.host_cache,
        options.max_positions(),
        request.max_tensor_bytes,
    )?;
    if host_options.mode != ferrule_model::transformer::host_experts::ExpertPrewarmMode::Full {
        return Err(ep_unsupported(
            "Qwen3.5 EP requires full shared host prewarm",
        ));
    }
    let resources =
        resources.with_host_expert_prewarm(host_options, request.max_tensor_bytes, |progress| {
            tracing::info!(
                read_bytes = progress.read_bytes,
                total_bytes = progress.total_bytes,
                completed_parameters = progress.completed_parameters,
                total_parameters = progress.total_parameters,
                elapsed_seconds = progress.elapsed.as_secs_f64(),
                "Qwen3.5 EP single host warm progress"
            );
        })?;
    let host = resources
        .host_experts()
        .cloned()
        .ok_or_else(|| ep_unsupported("EP host image is not ready"))?;
    tracing::info!(
        generation=host.generation(), stats=?host.stats(),
        host_read_bytes=host.stats().io.bytes_read,
        host_hits=host.hits(),
        "Qwen3.5 EP shared host image complete before owner startup"
    );
    // Recheck after the potentially lengthy NAS warm, before ANY weight upload.
    check_physical(&physical, &device)?;
    let source_rank = owners.group.source_rank;
    let mut executor = ExpertParallelExecutor::new_numeric_f32(
        owners.group,
        Arc::new(resources.clone()),
        Arc::clone(&host),
        placement.expert_devices().to_vec(),
        NumericExpertConfig {
            max_parameter_bytes: request.max_tensor_bytes,
            max_device_bytes: owner_limit,
            workspace_bytes: OWNER_WORKSPACE,
            dispatch_timeout: DISPATCH_TIMEOUT,
        },
    )?;
    let ready = (|| -> Result<Vec<ExpertOwnerStats>> {
        let stats = executor.owner_stats()?;
        if stats.len() != placement.expert_parallel() {
            return Err(ep_unsupported("incomplete expert Ready barrier"));
        }

        for (slot, stats) in stats.iter().enumerate() {
            let cache = stats
                .cuda_cache
                .ok_or_else(|| ep_unsupported("numeric EP returned a non-CUDA owner"))?;
            tracing::info!(owner=stats.rank.owner.get(), ordinal=placement.expert_devices()[slot],
                owned_experts=stats.owned_experts.len(), cache=?cache, "Qwen3.5 EP owner ready");
            if stats.owned_experts.len() != owners.counts[slot]
                || cache.resident_experts != owners.counts[slot]
                || cache.resident_bytes != owners.weights[slot]
                || cache.evictions != 0
                || cache.pending_upload_bytes != 0
                || cache.quarantined
            {
                return Err(ep_unsupported(
                    "EP owned expert residency differs from admitted physical plan",
                ));
            }
        }
        Ok(stats)
    })();
    let baseline = finish_owner_admission(ready, || executor.shutdown().map_err(Error::from))?;
    let adapter = executor.into_hybrid_cuda(Rc::clone(device.operators()))?;
    #[cfg(test)]
    let test_device = device.clone();
    let runner = GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda_with_routed_experts(
        resources,
        tokenizer,
        options,
        device,
        HybridCudaRoutedExperts::new(
            source_rank,
            Box::new(JoinedExperts {
                adapter,
                baseline,
                devices: placement.expert_devices().to_vec(),
                source: source_rank,
                last_layer: request.max_layers - 1,
                host: Arc::clone(&host),
                ready_host_hits: host.hits(),
            }),
        ),
    )?;
    tracing::info!(resident_bytes=runner.forward_executor().module().resident_parameter_bytes(),
        root_budget=?root, physical_plan=?physical, host_generation=host.generation(),
        "Qwen3.5 EP root and all expert owners ready; root-only KV authority");
    Ok(PreparedRoot {
        runner,
        planes,
        accounting,
        host,
        limits,
        #[cfg(test)]
        device: test_device,
    })
}
