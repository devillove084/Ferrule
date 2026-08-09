//! Qwen3-MoE recipe for the model-independent decoder graph.

use std::sync::Arc;

use crate::nn::{
    DTypeConstraint, ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
};
use crate::support::TensorRole;
use crate::transformer::{
    Attention, DecoderLayer, DecoderModelParts, DecoderModelSpec, DecoderRecipe,
    DecoderRecipeError, Embedding, FeedForward, GqaAttention, Linear, Moe, MoeRouterSpec,
    NameMapper, Residual, RmsNorm, RotaryEmbedding, RotaryPairing, RotaryRegion, RotaryScaling,
    RouterScoreFunction, RouterSelection, StateDictSchema, StateDictSchemaBuilder,
};

use super::{Qwen3HfNameMapper, Qwen3MoeConfig};

#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen3MoeRecipe;

impl Qwen3MoeRecipe {
    pub const fn new() -> Self {
        Self
    }

    fn config(value: &serde_json::Value) -> Result<Qwen3MoeConfig, DecoderRecipeError> {
        Qwen3MoeConfig::from_value(value).map_err(DecoderRecipeError::config)
    }
}

impl DecoderRecipe for Qwen3MoeRecipe {
    fn build_spec(
        &self,
        value: &serde_json::Value,
    ) -> Result<DecoderModelSpec, DecoderRecipeError> {
        let config = Self::config(value)?;
        let rotary = RotaryEmbedding::new(
            config.head_dim,
            config.rope_theta,
            RotaryPairing::SplitHalf,
            RotaryRegion::Prefix {
                dimensions: config.head_dim,
            },
            RotaryScaling::None,
        )?;
        let router = MoeRouterSpec::new(
            config.num_experts,
            config.num_experts_per_tok,
            RouterScoreFunction::Softmax,
            RouterSelection::TopK,
            true,
            1.0,
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let attention = GqaAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                false,
                rotary.clone(),
            )?
            .with_qk_norms(
                RmsNorm::new(config.head_dim, config.rms_norm_eps)?,
                RmsNorm::new(config.head_dim, config.rms_norm_eps)?,
            )?;
            layers.push(DecoderLayer::new(
                layer,
                RmsNorm::new(config.hidden_size, config.rms_norm_eps)?,
                Attention::Gqa(attention),
                Residual::Add,
                RmsNorm::new(config.hidden_size, config.rms_norm_eps)?,
                FeedForward::Moe(Moe::new(
                    config.hidden_size,
                    config.moe_intermediate_size,
                    router.clone(),
                    false,
                )?),
                Residual::Add,
            )?);
        }
        Ok(DecoderModelSpec::new(DecoderModelParts {
            architecture: "Qwen3MoeForCausalLM".into(),
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
        value: &serde_json::Value,
    ) -> Result<StateDictSchema, DecoderRecipeError> {
        let spec = self.build_spec(value)?;
        let dtype = DTypeConstraint::exact(ParameterDType::Bf16);
        let mut builder = StateDictSchema::builder();
        let mut next_id = 1;
        let embedding = register(
            &mut builder,
            &mut next_id,
            "token_embedding.weight".into(),
            dtype.clone(),
            spec.token_embedding().weight_shape().into(),
            ParameterResidency::Static,
            TensorRole::TokenEmbedding,
            None,
        )?;
        register(
            &mut builder,
            &mut next_id,
            "final_norm.weight".into(),
            dtype.clone(),
            spec.final_norm().weight_shape().into(),
            ParameterResidency::Static,
            TensorRole::OutputNorm,
            None,
        )?;
        register(
            &mut builder,
            &mut next_id,
            "output.weight".into(),
            dtype.clone(),
            spec.output().weight_shape().into(),
            ParameterResidency::Static,
            TensorRole::OutputHead,
            spec.tie_word_embeddings().then_some(embedding),
        )?;
        for layer in spec.layers() {
            let index = layer.index();
            let residency = ParameterResidency::layer(index);
            register(
                &mut builder,
                &mut next_id,
                format!("layers.{index}.input_norm.weight"),
                dtype.clone(),
                layer.input_norm().weight_shape().into(),
                residency.clone(),
                TensorRole::AttentionNorm,
                None,
            )?;
            register(
                &mut builder,
                &mut next_id,
                format!("layers.{index}.post_attention_norm.weight"),
                dtype.clone(),
                layer.post_attention_norm().weight_shape().into(),
                residency.clone(),
                TensorRole::FeedForwardNorm,
                None,
            )?;
            let Attention::Gqa(attention) = layer.attention() else {
                unreachable!("Qwen3 recipe only builds GQA")
            };
            for (name, role, shape) in [
                (
                    "query",
                    TensorRole::AttentionQuery,
                    attention.query().weight_shape().to_vec(),
                ),
                (
                    "key",
                    TensorRole::AttentionKey,
                    attention.key().weight_shape().to_vec(),
                ),
                (
                    "value",
                    TensorRole::AttentionValue,
                    attention.value().weight_shape().to_vec(),
                ),
                (
                    "output",
                    TensorRole::AttentionOutput,
                    attention.output().weight_shape().to_vec(),
                ),
                (
                    "query_norm",
                    TensorRole::AttentionQueryNorm,
                    vec![attention.head_dim()],
                ),
                (
                    "key_norm",
                    TensorRole::AttentionKeyNorm,
                    vec![attention.head_dim()],
                ),
            ] {
                register(
                    &mut builder,
                    &mut next_id,
                    format!("layers.{index}.attention.{name}.weight"),
                    dtype.clone(),
                    shape,
                    residency.clone(),
                    role,
                    None,
                )?;
            }
            let FeedForward::Moe(moe) = layer.feed_forward() else {
                unreachable!("Qwen3 recipe only builds MoE layers")
            };
            register(
                &mut builder,
                &mut next_id,
                format!("layers.{index}.router.weight"),
                dtype.clone(),
                moe.router().weight_shape().into(),
                residency,
                TensorRole::RouterLogits,
                None,
            )?;
            for expert in 0..moe.router_spec().num_experts() {
                for (name, role, shape) in [
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
                    register(
                        &mut builder,
                        &mut next_id,
                        format!("layers.{index}.experts.{expert}.{name}.weight"),
                        dtype.clone(),
                        shape.into(),
                        ParameterResidency::expert(index, expert),
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
        value: &serde_json::Value,
    ) -> Result<Arc<dyn NameMapper>, DecoderRecipeError> {
        let config = Self::config(value)?;
        Ok(Arc::new(
            Qwen3HfNameMapper::from_config(&config).map_err(DecoderRecipeError::name_mapper)?,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn register(
    builder: &mut StateDictSchemaBuilder,
    next_id: &mut u64,
    path: String,
    dtype: DTypeConstraint,
    shape: Vec<usize>,
    residency: ParameterResidency,
    role: TensorRole,
    alias: Option<ParameterId>,
) -> Result<ParameterId, DecoderRecipeError> {
    let id = ParameterId::new(*next_id);
    *next_id = next_id
        .checked_add(1)
        .ok_or_else(|| DecoderRecipeError::invalid_config("Qwen3 parameter ID overflow"))?;
    let mut parameter = ParameterSpec::new(id, ModulePath::new(path)?, dtype, shape, residency)?;
    if let Some(alias) = alias {
        parameter = parameter.with_alias(alias);
    }
    builder.register_with_role(parameter, role)?;
    Ok(id)
}
