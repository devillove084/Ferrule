//! Dense Qwen3 and Qwen3-MoE recipes for the model-independent decoder graph.

use std::sync::Arc;

use crate::nn::{
    DTypeConstraint, ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
};
use crate::support::TensorRole;
use crate::transformer::{
    Attention, DecoderLayer, DecoderModelParts, DecoderModelSpec, DecoderRecipe,
    DecoderRecipeError, Embedding, FeedForward, GqaAttention, Linear, Moe, MoeRouterSpec,
    NameMapper, Residual, RmsNorm, RotaryEmbedding, RotaryPairing, RotaryRegion, RotaryScaling,
    RouterScoreFunction, RouterSelection, StateDictSchema, StateDictSchemaBuilder, SwiGlu,
};

use super::{Qwen3DenseConfig, Qwen3HfNameMapper, Qwen3MoeConfig};

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
        let router = MoeRouterSpec::new(
            config.num_experts,
            config.num_experts_per_tok,
            RouterScoreFunction::Softmax,
            RouterSelection::TopK,
            true,
            1.0,
        )?;
        build_qwen_spec(
            "Qwen3MoeForCausalLM",
            config.num_hidden_layers,
            config.hidden_size,
            config.vocab_size,
            config.max_position_embeddings,
            config.tie_word_embeddings,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
            config.rope_theta,
            FeedForward::Moe(Moe::new(
                config.hidden_size,
                config.moe_intermediate_size,
                router,
                false,
            )?),
        )
    }

    fn build_schema(
        &self,
        value: &serde_json::Value,
    ) -> Result<StateDictSchema, DecoderRecipeError> {
        let spec = self.build_spec(value)?;
        build_qwen_schema(&spec)
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

#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen3DenseRecipe;

impl Qwen3DenseRecipe {
    pub const fn new() -> Self {
        Self
    }
}

impl DecoderRecipe for Qwen3DenseRecipe {
    fn build_spec(
        &self,
        value: &serde_json::Value,
    ) -> Result<DecoderModelSpec, DecoderRecipeError> {
        let config = Qwen3DenseConfig::from_value(value).map_err(DecoderRecipeError::config)?;
        build_qwen_spec(
            "Qwen3ForCausalLM",
            config.num_hidden_layers,
            config.hidden_size,
            config.vocab_size,
            config.max_position_embeddings,
            config.tie_word_embeddings,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
            config.rope_theta,
            FeedForward::SwiGlu(SwiGlu::new(
                config.hidden_size,
                config.intermediate_size,
                false,
            )?),
        )
    }

    fn build_schema(
        &self,
        value: &serde_json::Value,
    ) -> Result<StateDictSchema, DecoderRecipeError> {
        build_qwen_schema(&self.build_spec(value)?)
    }

    fn build_name_mapper(
        &self,
        value: &serde_json::Value,
    ) -> Result<Arc<dyn NameMapper>, DecoderRecipeError> {
        let config = Qwen3DenseConfig::from_value(value).map_err(DecoderRecipeError::config)?;
        Ok(Arc::new(Qwen3HfNameMapper::from_dense_config(&config)))
    }
}

fn build_qwen_spec(
    architecture: &str,
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    max_positions: usize,
    tie_word_embeddings: bool,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    feed_forward: FeedForward,
) -> Result<DecoderModelSpec, DecoderRecipeError> {
    let rotary = RotaryEmbedding::new(
        head_dim,
        rope_theta,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix {
            dimensions: head_dim,
        },
        RotaryScaling::None,
    )?;
    let mut layers = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let attention = GqaAttention::new(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            false,
            rotary.clone(),
        )?
        .with_qk_norms(
            RmsNorm::new(head_dim, rms_norm_eps)?,
            RmsNorm::new(head_dim, rms_norm_eps)?,
        )?;
        layers.push(DecoderLayer::new(
            layer,
            RmsNorm::new(hidden_size, rms_norm_eps)?,
            Attention::Gqa(attention),
            Residual::Add,
            RmsNorm::new(hidden_size, rms_norm_eps)?,
            feed_forward.clone(),
            Residual::Add,
        )?);
    }
    Ok(DecoderModelSpec::new(DecoderModelParts {
        architecture: architecture.into(),
        hidden_size,
        vocab_size,
        max_sequence_length: Some(max_positions),
        token_embedding: Embedding::new(vocab_size, hidden_size, None)?,
        layers,
        final_norm: RmsNorm::new(hidden_size, rms_norm_eps)?,
        output: Linear::new(hidden_size, vocab_size, false)?,
        tie_word_embeddings,
    })?)
}

fn build_qwen_schema(spec: &DecoderModelSpec) -> Result<StateDictSchema, DecoderRecipeError> {
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
        if let FeedForward::SwiGlu(dense) = layer.feed_forward() {
            for (name, role, shape) in [
                (
                    "gate",
                    TensorRole::DenseMlpGate,
                    dense.gate().weight_shape(),
                ),
                ("up", TensorRole::DenseMlpUp, dense.up().weight_shape()),
                (
                    "down",
                    TensorRole::DenseMlpDown,
                    dense.down().weight_shape(),
                ),
            ] {
                register(
                    &mut builder,
                    &mut next_id,
                    format!("layers.{index}.feed_forward.{name}.weight"),
                    dtype.clone(),
                    shape.into(),
                    residency.clone(),
                    role,
                    None,
                )?;
            }
            continue;
        }
        let FeedForward::Moe(moe) = layer.feed_forward() else {
            unreachable!("Qwen3 feed-forward must be SwiGLU or MoE")
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
