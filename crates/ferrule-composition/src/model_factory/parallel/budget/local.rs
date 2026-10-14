//! Authoritative bound-resource resident admission, separate from schema estimates.
use crate::model_factory::invalid;
use crate::{Error, Result};
use ferrule_model::{decoder::GenericDecoderOptions, transformer::BoundDecoderResources};

#[derive(Clone, Copy)]
pub(in crate::model_factory) struct ResidentCapacityTarget {
    pub(in crate::model_factory) kv_pages: usize,
    pub(in crate::model_factory) kv_bytes: usize,
    pub(in crate::model_factory) state_bytes: usize,
    pub(in crate::model_factory) per_sequence_state_bytes: usize,
    pub(in crate::model_factory) total_device_bytes: usize,
}
#[derive(Debug)]
pub(in crate::model_factory) struct BoundMemoryPlan {
    pub(in crate::model_factory) options: GenericDecoderOptions,
    pub(in crate::model_factory) estimate: ferrule_model::decoder::HybridMemoryPlan,
    pub(in crate::model_factory) budget: ferrule_model::decoder::HybridReservationRequirements,
    pub(in crate::model_factory) total_device_bytes: usize,
}
impl BoundMemoryPlan {
    pub(in crate::model_factory) fn new(
        target: ResidentCapacityTarget,
        inputs: &super::super::inputs::NumericProviderInputs,
        resources: &BoundDecoderResources,
        max_tensor_bytes: u64,
    ) -> Result<Self> {
        use ferrule_model::decoder::{HybridMemoryPlan, HybridReservationRequirements};
        use ferrule_model::nn::ParameterResidency;
        let limits = inputs.limits;
        // Both the untouched embedding and head must pass the source reader's
        // bound. This does not authorize any persistent decoded expert cache.
        let largest = resources
            .state_dict()
            .parameters()
            .iter()
            .try_fold(0u64, |n, p| {
                let bytes = p
                    .weight()
                    .slice()
                    .bytes
                    .checked_add(p.scale().map_or(0, |scale| scale.slice().bytes))
                    .ok_or_else(|| invalid("numeric provider parameter read size overflow"))?;
                Ok::<_, Error>(n.max(bytes))
            })?;
        if max_tensor_bytes < largest {
            return Err(invalid(format!(
                "numeric provider max tensor read budget {max_tensor_bytes} is smaller than required {largest} bytes"
            )));
        }
        let cache = limits.expert_cache_policy()?;
        let options = super::decoder_options(
            resources,
            inputs.family.clone(),
            limits,
            max_tensor_bytes,
            inputs.numeric_profile,
        )?;
        let estimate = HybridMemoryPlan::for_resources(resources, &options, target.kv_pages)?;
        if estimate.per_sequence_state_bytes != target.per_sequence_state_bytes
            || estimate.kv_bytes != target.kv_bytes
        {
            return Err(invalid(
                "numeric provider model/runtime state or KV geometry mismatch",
            ));
        }
        // The model includes numeric scratch in workspace, but its reservation
        // already belongs to max_bytes. Do not charge a second cache or scratch.
        let workspace_cap = limits
            .workspace_bytes
            .checked_add(limits.scratch_bytes)
            .ok_or_else(|| invalid("numeric provider workspace plus numeric scratch overflow"))?;
        if estimate.workspace_bytes_upper_bound > workspace_cap {
            return Err(invalid(format!(
                "numeric provider model workspace {} exceeds workspace plus scratch cap {workspace_cap}",
                estimate.workspace_bytes_upper_bound
            )));
        }
        // The actual model estimator is authoritative, including its current
        // binding/storage accounting. Never substitute the schema target.
        let budget = HybridReservationRequirements {
            kv_pages: target.kv_pages,
            state_bytes: target.state_bytes,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: workspace_cap,
        };
        let total_device_bytes = [
            budget.weight_bytes,
            budget.state_bytes,
            budget.workspace_bytes,
            estimate.kv_bytes,
            limits.allocator_margin_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |n, bytes| {
            n.checked_add(bytes)
                .ok_or_else(|| invalid("numeric provider model admission total overflow"))
        })?;
        if total_device_bytes > limits.device_bytes {
            return Err(invalid(format!(
                "numeric provider model admission requires {total_device_bytes} bytes including margin, device budget {} (schema target {} is not authoritative)",
                limits.device_bytes, target.total_device_bytes
            )));
        }
        // Report the directory size separately; it is not a residency count.
        let routed_parameters = resources
            .state_dict()
            .parameters()
            .iter()
            .filter(|p| matches!(p.residency(), ParameterResidency::Expert { .. }))
            .count();
        tracing::info!(
            profile = ?inputs.numeric_profile,
            ?cache,
            numeric_scratch_bytes = limits.scratch_bytes,
            ?estimate,
            ?budget,
            total_device_bytes,
            routed_parameters,
            schema_target_bytes = target.total_device_bytes,
            "numeric provider compressed FP8 / F32 TF32x3 bounded model admission"
        );
        Ok(Self {
            options,
            estimate,
            budget,
            total_device_bytes,
        })
    }

    pub(in crate::model_factory) fn check_available_device_bytes(
        &self,
        free_bytes: usize,
    ) -> Result<()> {
        if self.total_device_bytes > free_bytes {
            return Err(invalid(format!(
                "numeric provider model admission requires {} bytes including margin, only {free_bytes} free",
                self.total_device_bytes
            )));
        }
        Ok(())
    }
}
