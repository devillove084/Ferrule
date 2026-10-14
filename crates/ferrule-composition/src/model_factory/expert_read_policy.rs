//! Opt-in host read policy. Validation never creates a worker or device owner.
use super::*;

fn invalid(message: &str) -> Error {
    Error::InvalidRequest {
        message: format!("expert read pipeline: {message}"),
    }
}

pub(super) fn validate_request(
    backend: BackendSelection,
    options: &ModelFactoryOptions,
    serving: &ServingModelOptions,
    numeric_model: bool,
) -> Result<()> {
    let Some(read) = serving.expert_read_pipeline else {
        return Ok(());
    };
    validate_bounds(read)?;
    if backend != BackendSelection::Cuda
        || (!numeric_model && serving.model_profile.is_none())
        || serving.parallel.is_some()
        || serving.engine == ServingEngineSelection::Pipeline
        || serving.kv_execution_policy != ferrule_runtime::engine::KvExecutionPolicy::Resident
        || options.qwen35_host_cache.is_some()
        || options.numeric_fp8_profile.is_some()
        || options.moe_hotset_experts != 0
        || serving.expert_host_cache_entries.is_some()
        || serving.expert_host_cache_mebibytes.is_some()
        || serving.expert_pinned_cache_entries.is_some()
        || serving.expert_pinned_cache_mebibytes.is_some()
        || serving.moe_hotset_experts.is_some()
    {
        return Err(invalid(
            "requires an explicit CUDA lazy unsharded numeric model profile; no CPU/dense/full-prewarm/parallel/cache/hotset fallback",
        ));
    }
    Ok(())
}

pub(super) fn validate_resolved(request: &ResolvedModelRequest) -> Result<()> {
    let Some(read) = request.execution.expert_read_pipeline else {
        return Ok(());
    };
    validate_bounds(read)?;
    if request.backend != ModelExecutionBackend::Cuda
        || request.execution.numeric_bf16_precision.is_none()
        || request.execution.support.storage != StorageEncoding::NumericFp8E4m3fnBf16Scales
        || request.execution.construction.kind() != SelectedImplementationKind::Resident
        || request.pipeline.is_some()
        || request.execution.kv_execution_policy
            != ferrule_runtime::engine::KvExecutionPolicy::Resident
    {
        return Err(invalid(
            "selected execution does not consume lazy unsharded numeric expert reads",
        ));
    }
    Ok(())
}

/// Conservative shape/source admission for every possible routed selection. This
/// describes host staging only; no compressed payload or device allocation occurs.
pub(super) fn validate_sources(
    resources: &ferrule_model::transformer::BoundDecoderResources,
    runner: &ferrule_model::decoder::GenericDecoderOptions,
    read: ferrule_model::transformer::ExpertReadPipelineOptions,
) -> Result<()> {
    use ferrule_model::transformer::FeedForward;
    use ferrule_model::{checkpoint::CheckpointTensorReader, nn::ParameterResidency};
    use std::collections::BTreeMap;
    validate_bounds(read)?;
    if resources.host_experts().is_some() {
        return Err(invalid("full host prewarm is not a read pipeline"));
    }
    validate_available_host(read)?;
    let reader = CheckpointTensorReader::new(runner.max_parameter_bytes());
    let payload_cap =
        read.max_host_bytes - read.max_metadata_bytes - read.worker_stack_bytes as u64;
    let mut experts = BTreeMap::<(usize, usize), (u64, u64, usize)>::new();
    for p in resources.state_dict().parameters() {
        let ParameterResidency::Expert { layer, expert } = *p.residency() else {
            continue;
        };
        if p.id() != p.canonical_id() {
            return Err(invalid("requires canonical individual expert projections"));
        }
        let source = p.numeric_fp8_source(
            p.numeric_fp8_encoding()
                .ok_or_else(|| invalid("requires paired numeric FP8 experts"))?,
        )?;
        if source.expert_count().is_some() {
            return Err(invalid("packed/sharded expert bindings are unsupported"));
        }
        let [rows, columns] = source.matrix_shape();
        let plan = source.plan_tile(&reader, None, 0..rows, 0..columns)?;
        let (bytes, metadata, projections) = experts.entry((layer, expert)).or_insert((0, 4096, 0));
        *bytes = bytes
            .checked_add(plan.read_plan().storage_bytes())
            .ok_or_else(|| invalid("whole expert payload overflows"))?;
        // Covers retained source/binding/read-plan clones and map/ticket nodes.
        // This intentionally overcharges the executor's current metadata ledger.
        let mut variable = 0u64;
        for slice in [source.weight(), source.scale()] {
            variable = variable
                .checked_add(slice.path.as_os_str().len() as u64)
                .and_then(|n| n.checked_add(slice.name.len() as u64))
                .ok_or_else(|| invalid("source metadata overflows"))?;
        }
        for extent in plan.read_plan().extents() {
            variable = variable
                .checked_add(extent.path().as_os_str().len() as u64)
                .and_then(|n| n.checked_add(256))
                .ok_or_else(|| invalid("extent metadata overflows"))?;
        }
        *metadata = variable
            .checked_mul(32)
            .and_then(|n| n.checked_add(8192))
            .and_then(|n| n.checked_add(*metadata))
            .ok_or_else(|| invalid("expert metadata overflows"))?;
        *projections += 1;
    }
    if experts.is_empty() {
        return Err(invalid("selected source has no routed numeric experts"));
    }
    for layer in resources.spec().layers() {
        let FeedForward::Moe(moe) = layer.feed_forward() else {
            continue;
        };
        let selected = runner
            .capabilities()
            .max_batch_tokens
            .checked_mul(moe.router_spec().experts_per_token())
            .ok_or_else(|| invalid("route capacity overflows"))?
            .min(moe.router_spec().num_experts());
        let mut metadata = Vec::new();
        for expert in 0..moe.router_spec().num_experts() {
            let &(bytes, bound, projections) = experts
                .get(&(layer.index(), expert))
                .ok_or_else(|| invalid("incomplete expert source catalog"))?;
            if projections != 3 {
                return Err(invalid("requires exactly gate/up/down expert projections"));
            }
            if bytes > runner.max_parameter_bytes() {
                return Err(invalid("whole expert exceeds checkpoint reader batch cap"));
            }
            if bytes.checked_mul(2).is_none_or(|n| n > payload_cap) {
                return Err(invalid(
                    "two whole expert payloads exceed explicit host budget; device cache bytes cannot fund host staging",
                ));
            }
            metadata.push(bound);
        }
        metadata.sort_unstable_by(|a, b| b.cmp(a));
        let required = metadata
            .into_iter()
            .take(selected)
            .try_fold(4096u64, |n, b| {
                n.checked_add(b)
                    .ok_or_else(|| invalid("selected metadata overflows"))
            })?;
        if required > read.max_metadata_bytes {
            return Err(invalid(
                "worst-case routed metadata exceeds explicit metadata budget",
            ));
        }
    }
    Ok(())
}

fn validate_bounds(read: ferrule_model::transformer::ExpertReadPipelineOptions) -> Result<()> {
    read.validate()?;
    if read.max_host_bytes > isize::MAX as u64 {
        return Err(invalid("host staging budget exceeds address space"));
    }
    Ok(())
}

pub(super) fn validate_available_host(
    read: ferrule_model::transformer::ExpertReadPipelineOptions,
) -> Result<()> {
    validate_bounds(read)?;
    let available = host_memory::snapshot()?.available_bytes();
    if read.max_host_bytes > available {
        return Err(invalid(
            "explicit host staging budget exceeds currently available host/cgroup memory; device cache bytes cannot fund it",
        ));
    }
    Ok(())
}
