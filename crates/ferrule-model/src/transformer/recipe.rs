use std::sync::Arc;

use serde::Deserialize;
use thiserror::Error;

use crate::checkpoint::HfSafetensorsInventory;
use crate::nn::{
    DTypeConstraint, ModulePath, ModulePathError, ParameterDType, ParameterId, ParameterResidency,
    ParameterSpec, ParameterSpecError,
};
use crate::support::TensorRole;

use super::{
    Attention, DecoderLayer, DecoderModelParts, DecoderModelSpec, DescriptorError, Embedding,
    ExactNameMapper, FeedForward, GqaAttention, Linear, Moe, MoeRouterSpec, NameMapError,
    NameMapper, NameMapping, Residual, RmsNorm, RotaryEmbedding, RotaryPairing, RotaryRegion,
    RotaryScaling, RouterScoreFunction, RouterSelection, StateDictSchema, StateDictSchemaError,
};

/// Products of translating one decoder configuration into inference metadata.
pub struct DecoderRecipeOutput {
    spec: DecoderModelSpec,
    schema: StateDictSchema,
    name_mapper: Arc<dyn NameMapper>,
}

impl DecoderRecipeOutput {
    pub fn new(
        spec: DecoderModelSpec,
        schema: StateDictSchema,
        name_mapper: Arc<dyn NameMapper>,
    ) -> Self {
        Self {
            spec,
            schema,
            name_mapper,
        }
    }

    pub fn spec(&self) -> &DecoderModelSpec {
        &self.spec
    }

    pub fn schema(&self) -> &StateDictSchema {
        &self.schema
    }

    pub fn name_mapper(&self) -> &dyn NameMapper {
        self.name_mapper.as_ref()
    }

    pub fn into_parts(self) -> (DecoderModelSpec, StateDictSchema, Arc<dyn NameMapper>) {
        (self.spec, self.schema, self.name_mapper)
    }
}

/// Family adapter that translates configuration only.
///
/// Recipes do not open checkpoints, bind tensors, materialize weights, or expose
/// execution/training behavior. Keeping this interface object-safe allows model
/// registries to select a recipe without parameterizing the decoder type.
pub trait DecoderRecipe: Send + Sync {
    fn build_spec(
        &self,
        config: &serde_json::Value,
    ) -> Result<DecoderModelSpec, DecoderRecipeError>;

    fn build_schema(
        &self,
        config: &serde_json::Value,
    ) -> Result<StateDictSchema, DecoderRecipeError>;

    fn build_name_mapper(
        &self,
        config: &serde_json::Value,
    ) -> Result<Arc<dyn NameMapper>, DecoderRecipeError>;

    fn build_spec_with_inventory(
        &self,
        config: &serde_json::Value,
        _inventory: &HfSafetensorsInventory,
    ) -> Result<DecoderModelSpec, DecoderRecipeError> {
        self.build_spec(config)
    }

    fn build_schema_with_inventory(
        &self,
        config: &serde_json::Value,
        _inventory: &HfSafetensorsInventory,
    ) -> Result<StateDictSchema, DecoderRecipeError> {
        self.build_schema(config)
    }

    fn build_name_mapper_with_inventory(
        &self,
        config: &serde_json::Value,
        _inventory: &HfSafetensorsInventory,
    ) -> Result<Arc<dyn NameMapper>, DecoderRecipeError> {
        self.build_name_mapper(config)
    }

    fn build(&self, config: &serde_json::Value) -> Result<DecoderRecipeOutput, DecoderRecipeError> {
        Ok(DecoderRecipeOutput::new(
            self.build_spec(config)?,
            self.build_schema(config)?,
            self.build_name_mapper(config)?,
        ))
    }

    fn build_with_inventory(
        &self,
        config: &serde_json::Value,
        inventory: &HfSafetensorsInventory,
    ) -> Result<DecoderRecipeOutput, DecoderRecipeError> {
        Ok(DecoderRecipeOutput::new(
            self.build_spec_with_inventory(config, inventory)?,
            self.build_schema_with_inventory(config, inventory)?,
            self.build_name_mapper_with_inventory(config, inventory)?,
        ))
    }
}

#[derive(Debug, Error)]
pub enum DecoderRecipeError {
    #[error("invalid decoder config: {message}")]
    InvalidConfig { message: String },
    #[error("invalid decoder config: {source}")]
    Config {
        #[source]
        source: ferrule_common::Error,
    },
    #[error(transparent)]
    Descriptor(#[from] DescriptorError),
    #[error(transparent)]
    Schema(#[from] StateDictSchemaError),
    #[error(transparent)]
    Parameter(#[from] ParameterSpecError),
    #[error(transparent)]
    Path(#[from] ModulePathError),
    #[error("cannot construct decoder name mapper: {source}")]
    NameMapper {
        #[source]
        source: NameMapError,
    },
}

impl DecoderRecipeError {
    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }

    pub fn config(source: ferrule_common::Error) -> Self {
        Self::Config { source }
    }

    pub fn name_mapper(source: NameMapError) -> Self {
        Self::NameMapper { source }
    }
}

/// Family-neutral two-layer GQA+MoE recipe used to validate replacement APIs.
///
/// External tensor names are `synthetic.<canonical-path>`. The recipe performs
/// no I/O and materializes no payloads.
#[derive(Debug, Clone, Copy, Default)]
pub struct SyntheticDecoderRecipe;

impl SyntheticDecoderRecipe {
    pub const fn new() -> Self {
        Self
    }

    pub fn external_name(path: &ModulePath) -> String {
        format!("synthetic.{path}")
    }

    fn config(config: &serde_json::Value) -> Result<SyntheticDecoderConfig, DecoderRecipeError> {
        serde_json::from_value(config.clone()).map_err(|source| DecoderRecipeError::Config {
            source: ferrule_common::Error::ModelSource {
                source: Box::new(source),
            },
        })
    }
}

impl DecoderRecipe for SyntheticDecoderRecipe {
    fn build_spec(
        &self,
        config: &serde_json::Value,
    ) -> Result<DecoderModelSpec, DecoderRecipeError> {
        let config = Self::config(config)?;
        let rotary = RotaryEmbedding::new(
            config.head_dim,
            config.rope_theta,
            RotaryPairing::Interleaved,
            RotaryRegion::Prefix {
                dimensions: config.head_dim,
            },
            RotaryScaling::None,
        )?;
        let router = MoeRouterSpec::new(
            config.num_experts,
            config.experts_per_token,
            RouterScoreFunction::Softmax,
            RouterSelection::TopK,
            true,
            1.0,
        )?;
        let mut layers = Vec::with_capacity(2);
        for index in 0..2 {
            layers.push(DecoderLayer::new(
                index,
                RmsNorm::new(config.hidden_size, config.rms_norm_eps)?,
                Attention::Gqa(GqaAttention::new(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    false,
                    rotary.clone(),
                )?),
                Residual::Add,
                RmsNorm::new(config.hidden_size, config.rms_norm_eps)?,
                FeedForward::Moe(Moe::new(
                    config.hidden_size,
                    config.intermediate_size,
                    router.clone(),
                    false,
                )?),
                Residual::Add,
            )?);
        }
        Ok(DecoderModelSpec::new(DecoderModelParts {
            architecture: "synthetic-decoder".into(),
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            max_sequence_length: Some(config.max_position_embeddings),
            token_embedding: Embedding::new(config.vocab_size, config.hidden_size, None)?,
            layers,
            final_norm: RmsNorm::new(config.hidden_size, config.rms_norm_eps)?,
            output: Linear::new(config.hidden_size, config.vocab_size, false)?,
            tie_word_embeddings: config.tie_word_embeddings,
        })?)
    }

    fn build_schema(
        &self,
        config: &serde_json::Value,
    ) -> Result<StateDictSchema, DecoderRecipeError> {
        let spec = self.build_spec(config)?;
        let dtype = DTypeConstraint::one_of([ParameterDType::F32, ParameterDType::Bf16])?;
        let mut builder = StateDictSchema::builder();
        let mut next_id = 1u64;
        let embedding_id = register_parameter(
            &mut builder,
            &mut next_id,
            "token_embedding.weight",
            dtype.clone(),
            spec.token_embedding().weight_shape(),
            ParameterResidency::Static,
            TensorRole::TokenEmbedding,
            None,
        )?;
        register_parameter(
            &mut builder,
            &mut next_id,
            "final_norm.weight",
            dtype.clone(),
            spec.final_norm().weight_shape(),
            ParameterResidency::Static,
            TensorRole::OutputNorm,
            None,
        )?;
        register_parameter(
            &mut builder,
            &mut next_id,
            "output.weight",
            dtype.clone(),
            spec.output().weight_shape(),
            ParameterResidency::Static,
            TensorRole::OutputHead,
            spec.tie_word_embeddings().then_some(embedding_id),
        )?;

        for layer in spec.layers() {
            let layer_index = layer.index();
            let residency = ParameterResidency::layer(layer_index);
            register_parameter(
                &mut builder,
                &mut next_id,
                format!("layers.{layer_index}.input_norm.weight"),
                dtype.clone(),
                layer.input_norm().weight_shape(),
                residency.clone(),
                TensorRole::AttentionNorm,
                None,
            )?;
            register_parameter(
                &mut builder,
                &mut next_id,
                format!("layers.{layer_index}.post_attention_norm.weight"),
                dtype.clone(),
                layer.post_attention_norm().weight_shape(),
                residency.clone(),
                TensorRole::FeedForwardNorm,
                None,
            )?;
            let Attention::Gqa(attention) = layer.attention() else {
                return Err(DecoderRecipeError::invalid_config(
                    "synthetic decoder unexpectedly lost GQA",
                ));
            };
            for (suffix, role, shape) in [
                (
                    "query",
                    TensorRole::AttentionQuery,
                    attention.query().weight_shape(),
                ),
                (
                    "key",
                    TensorRole::AttentionKey,
                    attention.key().weight_shape(),
                ),
                (
                    "value",
                    TensorRole::AttentionValue,
                    attention.value().weight_shape(),
                ),
                (
                    "output",
                    TensorRole::AttentionOutput,
                    attention.output().weight_shape(),
                ),
            ] {
                register_parameter(
                    &mut builder,
                    &mut next_id,
                    format!("layers.{layer_index}.attention.{suffix}.weight"),
                    dtype.clone(),
                    shape,
                    residency.clone(),
                    role,
                    None,
                )?;
            }
            let FeedForward::Moe(moe) = layer.feed_forward() else {
                return Err(DecoderRecipeError::invalid_config(
                    "synthetic decoder unexpectedly lost MoE",
                ));
            };
            register_parameter(
                &mut builder,
                &mut next_id,
                format!("layers.{layer_index}.router.weight"),
                dtype.clone(),
                moe.router().weight_shape(),
                residency,
                TensorRole::RouterLogits,
                None,
            )?;
            for expert in 0..moe.router_spec().num_experts() {
                let expert_residency = ParameterResidency::expert(layer_index, expert);
                for (suffix, role, shape) in [
                    (
                        "gate",
                        TensorRole::RoutedExpertGate,
                        moe.expert().gate().weight_shape(),
                    ),
                    (
                        "up",
                        TensorRole::RoutedExpertUp,
                        moe.expert().up().weight_shape(),
                    ),
                    (
                        "down",
                        TensorRole::RoutedExpertDown,
                        moe.expert().down().weight_shape(),
                    ),
                ] {
                    register_parameter(
                        &mut builder,
                        &mut next_id,
                        format!("layers.{layer_index}.experts.{expert}.{suffix}.weight"),
                        dtype.clone(),
                        shape,
                        expert_residency.clone(),
                        role,
                        None,
                    )?;
                }
            }
        }
        Ok(builder.build()?)
    }

    fn build_name_mapper(
        &self,
        config: &serde_json::Value,
    ) -> Result<Arc<dyn NameMapper>, DecoderRecipeError> {
        let schema = self.build_schema(config)?;
        let mut mapper = ExactNameMapper::new();
        for parameter in schema
            .parameters()
            .iter()
            .filter(|parameter| parameter.alias_of().is_none())
        {
            mapper
                .insert(
                    Self::external_name(parameter.path()),
                    NameMapping::weight(parameter.path().clone()),
                )
                .map_err(DecoderRecipeError::name_mapper)?;
        }
        Ok(Arc::new(mapper))
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SyntheticDecoderConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    num_experts: usize,
    experts_per_token: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    #[serde(default)]
    tie_word_embeddings: bool,
}

#[allow(clippy::too_many_arguments)]
fn register_parameter(
    builder: &mut super::StateDictSchemaBuilder,
    next_id: &mut u64,
    path: impl Into<String>,
    dtype: DTypeConstraint,
    shape: impl Into<Vec<usize>>,
    residency: ParameterResidency,
    role: TensorRole,
    alias_of: Option<ParameterId>,
) -> Result<ParameterId, DecoderRecipeError> {
    let id = ParameterId::new(*next_id);
    *next_id = next_id
        .checked_add(1)
        .ok_or_else(|| DecoderRecipeError::invalid_config("synthetic parameter ID overflow"))?;
    let mut parameter = ParameterSpec::new(id, ModulePath::new(path)?, dtype, shape, residency)?;
    if let Some(alias_of) = alias_of {
        parameter = parameter.with_alias(alias_of);
    }
    builder.register_with_role(parameter, role)?;
    Ok(id)
}
