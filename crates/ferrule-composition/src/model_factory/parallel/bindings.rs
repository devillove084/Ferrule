//! Checks on already-bound resources. Family recipes and support guards stay in adapters.
use super::*;
pub(in crate::model_factory) fn validate_bound(
    request: &ResolvedModelRequest,
    resources: &BoundDecoderResources,
) -> Result<()> {
    let spec = resources.spec();
    if request.max_layers != spec.layers().len()
        || !spec.attachments().is_empty()
        || resources.host_experts().is_some()
        || spec
            .max_sequence_length()
            .is_some_and(|n| request.driver_config.ctx_size > n)
        || !resources.state_dict().validate_source_identities()
    {
        return Err(invalid(
            "numeric provider requires fresh full-depth text-only checkpoint binding",
        ));
    }
    resources.validate_parameter_limits(request.max_tensor_bytes, request.max_tensor_bytes)?;
    Ok(())
}

pub(in crate::model_factory) fn validate_cache_options(
    request: &ResolvedModelRequest,
) -> Result<()> {
    crate::model_factory::execution_policy::validate_defaults(&[
        (
            "expert cache",
            request.expert_memory_policy == ferrule_model::ExpertMemoryPolicy::default(),
        ),
        ("expert hotset", request.moe_hotset_experts == 0),
        (
            "expert reader",
            request.expert_reader_max_tensor_bytes == 64 << 20,
        ),
        ("output head", request.output_head_chunk_rows == 4096),
    ])
    .map_err(|e| invalid(e.to_string()))
}

pub(in crate::model_factory) fn validate_options(request: &ResolvedModelRequest) -> Result<()> {
    validate_cache_options(request)?;
    crate::model_factory::execution_policy::validate_defaults(&[
        (
            "native proposals",
            !request.driver_config.enable_native_proposals,
        ),
        (
            "prefix cache",
            request.scheduler_config.prefix_cache_capacity_pages == 0,
        ),
    ])
    .map_err(|e| invalid(e.to_string()))
}
