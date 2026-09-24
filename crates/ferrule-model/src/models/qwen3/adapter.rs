//! Thin dense/MoE Qwen3 composition over the generic CPU decoder runner.

use std::path::Path;

use ferrule_common::execution::{KvElementType, KvLayoutSchema};
use ferrule_common::{Error, Result};

use crate::decoder::{
    CpuPagedKvBackend, CpuPagedKvPool, GenericDecoderOptions, GenericDecoderRunner, PagedKvBackend,
    StandardGqaPlanes,
};
use crate::execution::ExecutionPrecisionPolicy;
use crate::{ModelExecutionBackend, ModelFamily, WeightSource};

use super::{Qwen3DenseConfig, Qwen3DenseRecipe, Qwen3MoeCheckpoint};
use crate::TokenizerHandle;
use crate::transformer::{BoundDecoderResources, DecoderLoadOptions, HFDecoderCheckpoint};

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
        checkpoint.resources().validate_parameter_limits(
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

fn model_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("Qwen3 adapter: {}", message.into()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3DensePrepareOptions {
    pub page_size: usize,
    pub max_dense_tensor_bytes: u64,
    pub max_layers: Option<usize>,
}

impl Default for Qwen3DensePrepareOptions {
    fn default() -> Self {
        Self {
            page_size: 16,
            max_dense_tensor_bytes: 1024 * 1024 * 1024,
            max_layers: None,
        }
    }
}

pub struct Qwen3DenseAdapter {
    config: Qwen3DenseConfig,
    checkpoint: HFDecoderCheckpoint,
    tokenizer: TokenizerHandle,
    kv_schema: StandardGqaPlanes,
    active_layers: usize,
    max_parameter_bytes: u64,
}

impl Qwen3DenseAdapter {
    pub fn load_hf_with_options(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: Qwen3DensePrepareOptions,
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
        options: Qwen3DensePrepareOptions,
        backend: ModelExecutionBackend,
    ) -> Result<Self> {
        if backend != ModelExecutionBackend::Cpu {
            return Err(model_error(
                "dense CUDA is unsupported until the standard decoder graph has a native CUDA adapter",
            ));
        }
        if max_tensor_bytes == 0 || options.page_size == 0 || options.max_dense_tensor_bytes == 0 {
            return Err(model_error("all Qwen3 load limits must be non-zero"));
        }
        let (config, checkpoint) = Self::bind_hf_metadata(model_dir)?;
        let active_layers = options.max_layers.unwrap_or(config.num_hidden_layers);
        if active_layers == 0 || active_layers > config.num_hidden_layers {
            return Err(model_error(format!(
                "max_layers must be in 1..={}, got {active_layers}",
                config.num_hidden_layers
            )));
        }
        let max_parameter_bytes = max_tensor_bytes.min(options.max_dense_tensor_bytes);
        checkpoint
            .resources()
            .validate_parameter_limits(max_parameter_bytes, max_parameter_bytes)?;
        let kv_schema = StandardGqaPlanes::new(
            active_layers,
            config.num_key_value_heads,
            config.head_dim,
            options.page_size,
            config.max_position_embeddings,
            KvElementType::Bf16,
        )?;
        let tokenizer = TokenizerHandle::load(model_dir)?;
        Ok(Self {
            config,
            checkpoint,
            tokenizer,
            kv_schema,
            active_layers,
            max_parameter_bytes,
        })
    }

    /// Strict config/index/header/state-dict binding, without tensor payload reads,
    /// tokenizer loading, backend selection, or a full-weight read budget. Callers
    /// must validate their actual read geometry before materializing weights.
    /// Resident loading uses this same binding followed by full-parameter limits.
    pub fn bind_hf_metadata(model_dir: &Path) -> Result<(Qwen3DenseConfig, HFDecoderCheckpoint)> {
        let config_path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|error| {
            Error::context(
                format!("Qwen3 dense config '{}'", config_path.display()),
                error.into(),
            )
        })?;
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|source| Error::ModelSource {
                source: Box::new(source),
            })?;
        let config = Qwen3DenseConfig::from_value(&value)?;
        let checkpoint = DecoderLoadOptions::new(&Qwen3DenseRecipe::new(), &value)
            .open_hf_checkpoint(model_dir, ModelFamily::Qwen3)?;
        Ok((config, checkpoint))
    }

    /// Delegate metadata-only TP preflight to the shared standard decoder plan.
    /// Actual bounded reads still revalidate checkpoint source identities.
    pub fn validate_tensor_read_limits(
        resources: &BoundDecoderResources,
        tensor: &crate::transformer::StandardTensorPlan,
        max_tensor_bytes: u64,
    ) -> Result<()> {
        tensor.validate_read_limits(resources, max_tensor_bytes)
    }

    pub const fn config(&self) -> &Qwen3DenseConfig {
        &self.config
    }

    pub const fn resources(&self) -> &BoundDecoderResources {
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
        if max_positions == 0 || max_positions > self.config.max_position_embeddings {
            return Err(model_error(
                "runtime context is outside the Qwen3 position range",
            ));
        }
        let options = GenericDecoderOptions::standard_cpu(
            self.resources().spec(),
            ModelFamily::Qwen3,
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
        GenericDecoderRunner::new(
            self.checkpoint.into_resources(),
            self.tokenizer,
            PagedKvBackend::new(pool),
            options,
        )
    }
}
