//! Thin CPU composition: strict metadata partition -> generic binder -> hybrid runner.
use std::path::Path;
use std::sync::Arc;

use super::{Qwen35Config, Qwen35Metadata, Qwen35Recipe, model_error};
use crate::decoder::{
    GenericDecoderOptions, GenericDecoderRunner, HybridCpuDecoder, HybridStateSchema,
};
use crate::execution::ExecutionPrecisionPolicy;
use crate::transformer::{BoundDecoderResources, StateDictBinder};
use crate::{
    CheckpointTensorSlice, ModelExecutionBackend, ModelFamily, TokenizerHandle, WeightSource,
};
use ferrule_common::{Error, Result};

pub type Qwen35CpuRunner = GenericDecoderRunner<HybridCpuDecoder>;

/// Execution is always full-depth, F32 CPU; storage retains the strict BF16/F32 schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35PrepareOptions {
    pub page_size: usize,
    pub max_parameter_bytes: u64,
}
impl Default for Qwen35PrepareOptions {
    fn default() -> Self {
        Self {
            page_size: 16,
            max_parameter_bytes: 1 << 30,
        }
    }
}

pub struct Qwen35Adapter {
    metadata: Qwen35Metadata,
    resources: BoundDecoderResources,
    tokenizer: TokenizerHandle,
    options: Qwen35PrepareOptions,
}
impl Qwen35Adapter {
    /// Validate every attachment, but bind only text slices. No payload/tokenizer reads.
    pub fn bind_hf_metadata(model_dir: &Path) -> Result<(Qwen35Metadata, BoundDecoderResources)> {
        Self::bind_metadata(model_dir, Qwen35Metadata::open_hf(model_dir)?)
    }

    fn bind_metadata(
        model_dir: &Path,
        metadata: Qwen35Metadata,
    ) -> Result<(Qwen35Metadata, BoundDecoderResources)> {
        let spec = Qwen35Recipe::spec(metadata.config()).map_err(source_error)?;
        let schema = Qwen35Recipe::schema(metadata.config()).map_err(source_error)?;
        let text = metadata.partition().text().iter().map(|&index| {
            CheckpointTensorSlice::from_hf_inventory(
                model_dir,
                &metadata.inventory().tensors[index],
            )
        });
        let state_dict = StateDictBinder::new(&schema, metadata.name_mapper())
            .bind_slices(text)
            .map_err(source_error)?;
        let resources = BoundDecoderResources::new(spec, Arc::new(state_dict))?;
        Ok((metadata, resources))
    }

    pub fn load_hf(model_dir: &Path) -> Result<Self> {
        Self::load_hf_with_options(model_dir, Qwen35PrepareOptions::default())
    }

    pub fn load_hf_with_options(model_dir: &Path, options: Qwen35PrepareOptions) -> Result<Self> {
        Self::load_hf_with_options_and_backend(model_dir, options, ModelExecutionBackend::Cpu)
    }

    pub fn load_hf_with_options_and_backend(
        model_dir: &Path,
        options: Qwen35PrepareOptions,
        backend: ModelExecutionBackend,
    ) -> Result<Self> {
        if backend != ModelExecutionBackend::Cpu {
            return Err(super::Qwen35Unsupported::Backend(backend).into());
        }
        if options.page_size == 0 || options.max_parameter_bytes == 0 {
            return Err(model_error(
                "page_size and max_parameter_bytes must be non-zero",
            ));
        }
        let (metadata, resources) = Self::bind_hf_metadata(model_dir)?;
        resources
            .validate_parameter_limits(options.max_parameter_bytes, options.max_parameter_bytes)?;
        let tokenizer = TokenizerHandle::load(model_dir)?;
        Ok(Self {
            metadata,
            resources,
            tokenizer,
            options,
        })
    }

    pub fn config(&self) -> &Qwen35Config {
        self.metadata.config()
    }
    pub const fn metadata(&self) -> &Qwen35Metadata {
        &self.metadata
    }
    pub const fn resources(&self) -> &BoundDecoderResources {
        &self.resources
    }
    pub const fn backend_name(&self) -> &'static str {
        "cpu"
    }
    pub const fn backend_profile(&self) -> &'static str {
        "cpu-hybrid-f32-qwen35-0.8b"
    }
    pub fn state_schema(&self) -> Result<HybridStateSchema> {
        HybridStateSchema::from_spec(self.resources.spec(), self.resources.spec().layers().len())
    }

    pub fn into_decoder(
        self,
        max_positions: usize,
        max_batch_tokens: usize,
        max_sequences: usize,
    ) -> Result<Qwen35CpuRunner> {
        let options = GenericDecoderOptions::standard_cpu(
            self.resources.spec(),
            ModelFamily::Qwen35,
            WeightSource::Safetensors,
            self.options.page_size,
            max_positions,
            max_batch_tokens,
            max_sequences,
            self.options.max_parameter_bytes,
            ExecutionPrecisionPolicy::f32(),
        )?;
        GenericDecoderRunner::<HybridCpuDecoder>::hybrid_cpu(
            self.resources,
            self.tokenizer,
            options,
        )
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

fn source_error(source: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::ModelSource {
        source: Box::new(source),
    }
}
