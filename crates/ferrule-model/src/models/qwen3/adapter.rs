//! Thin Qwen3-MoE composition over the generic CPU decoder runner.

use std::path::Path;

use ferrule_common::execution::{KvElementType, KvLayoutSchema};
use ferrule_common::{Error, Result};

use crate::decoder::{
    CpuPagedKvBackend, CpuPagedKvPool, GenericDecoderOptions, GenericDecoderRunner, PagedKvBackend,
    StandardGqaPlanes,
};
use crate::execution::ExecutionPrecisionPolicy;
use crate::nn::ParameterResidency;
use crate::{ModelExecutionBackend, ModelFamily, WeightSource};

use super::Qwen3MoeCheckpoint;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3MoePrepareOptions {
    pub page_size: usize,
    pub max_dense_tensor_bytes: u64,
    pub max_expert_tensor_bytes: u64,
    pub max_layers: Option<usize>,
}

impl Default for Qwen3MoePrepareOptions {
    fn default() -> Self {
        Self {
            page_size: 16,
            max_dense_tensor_bytes: 1024 * 1024 * 1024,
            max_expert_tensor_bytes: 256 * 1024 * 1024,
            max_layers: None,
        }
    }
}

pub struct Qwen3MoeAdapter {
    checkpoint: Qwen3MoeCheckpoint,
    kv_schema: StandardGqaPlanes,
    active_layers: usize,
    max_parameter_bytes: u64,
}

impl Qwen3MoeAdapter {
    pub fn load_hf_with_options(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: Qwen3MoePrepareOptions,
    ) -> Result<Self> {
        Self::load_hf_with_options_and_backend(
            model_dir,
            max_tensor_bytes,
            options,
            ModelExecutionBackend::Cpu,
        )
    }

    pub fn load_hf_with_options_and_backend(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: Qwen3MoePrepareOptions,
        backend: ModelExecutionBackend,
    ) -> Result<Self> {
        if backend != ModelExecutionBackend::Cpu {
            return Err(model_error(
                "CUDA is unsupported until the standard decoder graph has a native CUDA adapter",
            ));
        }
        if max_tensor_bytes == 0
            || options.page_size == 0
            || options.max_dense_tensor_bytes == 0
            || options.max_expert_tensor_bytes == 0
        {
            return Err(model_error("all Qwen3 load limits must be non-zero"));
        }
        let checkpoint = Qwen3MoeCheckpoint::load_hf(model_dir)?;
        let active_layers = options
            .max_layers
            .unwrap_or(checkpoint.config.num_hidden_layers);
        if active_layers == 0 || active_layers > checkpoint.config.num_hidden_layers {
            return Err(model_error(format!(
                "max_layers must be in 1..={}, got {active_layers}",
                checkpoint.config.num_hidden_layers
            )));
        }
        validate_parameter_limits(
            &checkpoint,
            max_tensor_bytes.min(options.max_dense_tensor_bytes),
            options.max_expert_tensor_bytes,
        )?;
        let kv_schema = StandardGqaPlanes::new(
            active_layers,
            checkpoint.config.num_key_value_heads,
            checkpoint.config.head_dim,
            options.page_size,
            checkpoint.config.max_position_embeddings,
            KvElementType::Bf16,
        )?;
        Ok(Self {
            checkpoint,
            kv_schema,
            active_layers,
            max_parameter_bytes: max_tensor_bytes.min(options.max_dense_tensor_bytes),
        })
    }

    pub const fn config(&self) -> &super::Qwen3MoeConfig {
        &self.checkpoint.config
    }

    pub const fn resources(&self) -> &crate::transformer::BoundDecoderResources {
        self.checkpoint.resources()
    }

    pub const fn kv_schema(&self) -> &StandardGqaPlanes {
        &self.kv_schema
    }

    pub const fn backend_name(&self) -> &'static str {
        "cpu"
    }

    pub fn into_decoder(
        self,
        max_positions: usize,
        max_batch_tokens: usize,
        max_sequences: usize,
    ) -> Result<GenericDecoderRunner<CpuPagedKvBackend>> {
        if max_positions == 0 || max_positions > self.checkpoint.config.max_position_embeddings {
            return Err(model_error(
                "runtime context is outside the Qwen3 position range",
            ));
        }
        let options = GenericDecoderOptions::standard_cpu(
            self.checkpoint.resources().spec(),
            ModelFamily::QwenMoe,
            WeightSource::Safetensors,
            self.kv_schema.page_size(),
            max_positions,
            max_batch_tokens,
            max_sequences,
            self.max_parameter_bytes,
            ExecutionPrecisionPolicy::bf16_compatibility(),
        )?
        .with_active_layers(self.active_layers)?;
        let pool = CpuPagedKvPool::from_strategy(&self.kv_schema, 1)?;
        let backend = PagedKvBackend::new(pool);
        let (resources, tokenizer) = self.checkpoint.into_runtime_parts();
        GenericDecoderRunner::new(resources, tokenizer, backend, options)
    }
}

fn validate_parameter_limits(
    checkpoint: &Qwen3MoeCheckpoint,
    dense_limit: u64,
    expert_limit: u64,
) -> Result<()> {
    for parameter in checkpoint.resources().state_dict().parameters() {
        let limit = if matches!(parameter.residency(), ParameterResidency::Expert { .. }) {
            expert_limit
        } else {
            dense_limit
        };
        if parameter.weight().slice().bytes > limit {
            return Err(model_error(format!(
                "parameter '{}' is {} bytes, above its {}-byte load limit",
                parameter.path(),
                parameter.weight().slice().bytes,
                limit
            )));
        }
    }
    Ok(())
}

fn model_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("Qwen3-MoE adapter: {}", message.into()),
    }
}
