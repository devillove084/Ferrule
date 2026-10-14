//! Named serving profiles lower to catalog inputs, never to a separate planner.
use super::*;

#[path = "mixed_tensor_serving.rs"]
pub(super) mod tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServingModelProfile {
    /// Experimental numeric storage with mixed BF16/F32 boundaries, not legacy W8.
    Qwen35MixedBf16,
}

/// CLI-only provenance that cannot be recovered from resolved cache defaults.
/// In particular explicit lazy is confirmation; explicit workers/full prewarm
/// are unsupported by the mixed profile even when their values equal defaults.
#[derive(Debug, Clone, Copy, Default)]
pub struct ServingOptionProvenance {
    pub expert_prewarm: Option<ferrule_model::transformer::host_experts::ExpertPrewarmMode>,
    pub expert_prewarm_workers: Option<usize>,
}

impl ServingModelOptions {
    /// Validate explicit serving policy before opening the model path. This does
    /// not bind metadata or establish family support, feature availability or
    /// device admission; catalog lowering performs those remaining checks.
    pub fn validate_before_model_io(
        &self,
        backend: BackendSelection,
        options: &ModelFactoryOptions,
        provenance: &ServingOptionProvenance,
    ) -> Result<()> {
        super::expert_read_policy::validate_request(backend, options, self, false)?;
        if let Some(profile) = self.model_profile {
            return profile.validate_request(backend, options, self, provenance);
        }
        if self.model_precision == Some(ModelPrecision::Bf16) {
            return bf16_serving::validate_dense_request(backend, options, self);
        }
        if self.kv_execution_policy != ferrule_runtime::engine::KvExecutionPolicy::Resident {
            return dense_f32_kv_policy::validate_request(backend, options, self);
        }
        if let Some(parallel) = &self.parallel {
            parallel
                .parallelism
                .validate()
                .map_err(|e| invalid(&e.to_string()))?;
        }
        Ok(())
    }
}

impl ServingModelProfile {
    pub(super) fn lower(self, serving: &mut ServingModelOptions) -> Result<NumericModelRequest> {
        serving.model_precision = Some(ModelPrecision::Bf16);
        Ok(NumericModelRequest {
            storage: NumericStoragePolicy::E4M3FnBf16Scales,
            precision: ferrule_model::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate,
        })
    }

    fn validate_request(
        self,
        backend: BackendSelection,
        options: &ModelFactoryOptions,
        serving: &ServingModelOptions,
        provenance: &ServingOptionProvenance,
    ) -> Result<()> {
        use execution_policy::{OverridePolicy, validate_override};
        validate_override(
            "model precision override",
            serving.model_precision.as_ref(),
            &ModelPrecision::Bf16,
            OverridePolicy::Unsupported,
        )
        .map_err(|_| {
            invalid("profile already selects BF16 model precision; omit the precision override")
        })?;
        if options.scheduler_config.max_batch_tokens > 32
            || options.scheduler_config.max_batch_tokens > options.driver_config.ctx_size
        {
            return Err(invalid(
                "requires batch <= min(32, context); no silent batch reduction",
            ));
        }
        if serving.parallel.is_some() {
            if provenance.expert_prewarm.is_some() || provenance.expert_prewarm_workers.is_some() {
                return Err(invalid(
                    "Mixed TP2 uses resident local expert shards; expert prewarm flags are unsupported",
                ));
            }
            let mut numeric = serving.clone();
            let request = self.lower(&mut numeric)?;
            tensor::validate_request(backend, options, &numeric, request.storage)?;
            return tensor::require_quality();
        }
        use ferrule_model::transformer::host_experts::ExpertPrewarmMode;
        validate_override(
            "expert prewarm",
            provenance.expert_prewarm.as_ref(),
            &ExpertPrewarmMode::Lazy,
            OverridePolicy::DefaultOnly,
        )
        .map_err(|_| invalid("Mixed requires lazy bounded experts; full prewarm is unsupported"))?;
        validate_override(
            "expert prewarm workers",
            provenance.expert_prewarm_workers.as_ref(),
            &0,
            OverridePolicy::Unsupported,
        )
        .map_err(|e| invalid(&e.to_string()))?;
        let limits = options.qwen35_moe_capacity.unwrap_or_default();
        if limits.device_bytes == 0
            || limits.workspace_bytes == 0
            || limits.max_experts == 0
            || limits.max_bytes <= limits.scratch_bytes
        {
            return Err(invalid(
                "Mixed requires nonzero device/workspace/cache capacities and expert cache bytes greater than scratch",
            ));
        }
        self.validate_startup_environment(std::env::var_os("CUBLAS_WORKSPACE_CONFIG").as_deref())?;
        let mut numeric = serving.clone();
        let request = self.lower(&mut numeric)?;
        bf16_serving::validate_numeric_request(
            backend,
            options,
            &numeric,
            request.storage,
            request.precision,
        )
    }

    /// No environment mutation and no CUDA initialization.
    pub fn validate_startup_environment(self, workspace: Option<&std::ffi::OsStr>) -> Result<()> {
        if workspace != Some(std::ffi::OsStr::new(":0:0")) {
            return Err(invalid(
                "requires literal CUBLAS_WORKSPACE_CONFIG=:0:0 before starting Ferrule; restart with CUBLAS_WORKSPACE_CONFIG=:0:0 ferrule serve ...; no late environment mutation or arithmetic fallback",
            ));
        }
        Ok(())
    }
}
fn invalid(message: &str) -> Error {
    Error::InvalidRequest {
        message: format!("experimental qwen35-mixed-bf16 serving: {message}"),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/mixed_serving.rs"]
mod tests;

#[cfg(test)]
pub(super) use tests::fixture;
