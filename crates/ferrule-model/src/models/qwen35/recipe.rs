//! Qwen3.5 descriptors and schema over the shared hybrid decoder, without execution code.
use std::sync::Arc;

use crate::TensorRole;
use crate::nn::{DTypeConstraint, ModulePath, ParameterId, ParameterResidency, ParameterSpec};
use crate::transformer::{
    Attention, DecoderLayer, DecoderModelParts, DecoderModelSpec, DecoderRecipe,
    DecoderRecipeError, Embedding, FeedForward, GatedDeltaNetAttention, GqaAttention, Linear,
    NameMapper, Residual, RmsNorm, RotaryEmbedding, RotaryPairing, RotaryRegion, RotaryScaling,
    StateDictSchema, SwiGlu,
};

use super::{Qwen35Config, Qwen35HfNameMapper, Qwen35LayerType, Qwen35TensorPartitionKind};

#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen35Recipe;

impl Qwen35Recipe {
    pub const fn new() -> Self {
        Self
    }

    pub fn spec(config: &Qwen35Config) -> Result<DecoderModelSpec, DecoderRecipeError> {
        let t = config.text();
        let norm = |width| RmsNorm::new(width, t.rms_norm_eps).map(RmsNorm::with_one_plus_weight);
        let rotary = RotaryEmbedding::new(
            t.head_dim,
            t.rope_parameters.rope_theta,
            RotaryPairing::SplitHalf,
            RotaryRegion::Prefix {
                dimensions: config.semantics().rotary_dimensions,
            },
            RotaryScaling::None,
        )?;
        let mut layers = Vec::with_capacity(t.num_hidden_layers);
        for (index, kind) in config.layer_types().iter().enumerate() {
            let attention = match kind {
                Qwen35LayerType::LinearAttention => {
                    Attention::GatedDeltaNet(GatedDeltaNetAttention::new(
                        t.hidden_size,
                        t.linear_num_key_heads,
                        t.linear_num_value_heads,
                        t.linear_key_head_dim,
                        t.linear_value_head_dim,
                        t.linear_conv_kernel_dim,
                        t.rms_norm_eps,
                        false,
                    )?)
                }
                Qwen35LayerType::FullAttention => Attention::Gqa(
                    GqaAttention::new(
                        t.hidden_size,
                        t.num_attention_heads,
                        t.num_key_value_heads,
                        t.head_dim,
                        false,
                        rotary.clone(),
                    )?
                    .with_gated_query()?
                    .with_qk_norms(norm(t.head_dim)?, norm(t.head_dim)?)?,
                ),
            };
            layers.push(DecoderLayer::new(
                index,
                norm(t.hidden_size)?,
                attention,
                Residual::Add,
                norm(t.hidden_size)?,
                FeedForward::SwiGlu(SwiGlu::new(t.hidden_size, t.intermediate_size, false)?),
                Residual::Add,
            )?);
        }
        Ok(DecoderModelSpec::new(DecoderModelParts {
            architecture: config.architecture().into(),
            hidden_size: t.hidden_size,
            vocab_size: t.vocab_size,
            max_sequence_length: Some(t.max_position_embeddings),
            token_embedding: Embedding::new(t.vocab_size, t.hidden_size, None)?,
            layers,
            final_norm: norm(t.hidden_size)?,
            output: Linear::new(t.hidden_size, t.vocab_size, false)?,
            tie_word_embeddings: t.tie_word_embeddings,
        })?)
    }

    pub fn schema(config: &Qwen35Config) -> Result<StateDictSchema, DecoderRecipeError> {
        let mapper = Qwen35HfNameMapper::new(config);
        let mut builder = StateDictSchema::builder();
        let mut embedding = None;
        let mut next = 1;
        for tensor in mapper
            .tensors()
            .filter(|s| s.partition == Qwen35TensorPartitionKind::Text)
        {
            let path = tensor
                .canonical_path
                .clone()
                .expect("text schema has canonical paths");
            let (residency, role) = parameter_role(path.as_str())?;
            let id = ParameterId::new(next);
            next += 1;
            if role == TensorRole::TokenEmbedding {
                embedding = Some(id);
            }
            builder.register_with_role(
                ParameterSpec::new(
                    id,
                    path,
                    DTypeConstraint::exact(tensor.dtype.clone()),
                    tensor.shape.clone(),
                    residency,
                )?,
                role,
            )?;
        }
        let t = config.text();
        builder.register_with_role(
            ParameterSpec::new(
                ParameterId::new(next),
                ModulePath::new(mapper.output_alias().0)?,
                DTypeConstraint::exact(crate::nn::ParameterDType::Bf16),
                vec![t.vocab_size, t.hidden_size],
                ParameterResidency::Static,
            )?
            .with_alias(embedding.expect("text schema includes embedding")),
            TensorRole::OutputHead,
        )?;
        Ok(builder.build()?)
    }
}

impl DecoderRecipe for Qwen35Recipe {
    fn build_spec(
        &self,
        value: &serde_json::Value,
    ) -> Result<DecoderModelSpec, DecoderRecipeError> {
        Self::spec(&Qwen35Config::from_value(value).map_err(DecoderRecipeError::config)?)
    }
    fn build_schema(
        &self,
        value: &serde_json::Value,
    ) -> Result<StateDictSchema, DecoderRecipeError> {
        Self::schema(&Qwen35Config::from_value(value).map_err(DecoderRecipeError::config)?)
    }
    fn build_name_mapper(
        &self,
        value: &serde_json::Value,
    ) -> Result<Arc<dyn NameMapper>, DecoderRecipeError> {
        let config = Qwen35Config::from_value(value).map_err(DecoderRecipeError::config)?;
        Ok(Arc::new(Qwen35HfNameMapper::new(&config)))
    }
}

fn parameter_role(path: &str) -> Result<(ParameterResidency, TensorRole), DecoderRecipeError> {
    let static_role = match path {
        "token_embedding.weight" => Some(TensorRole::TokenEmbedding),
        "final_norm.weight" => Some(TensorRole::OutputNorm),
        _ => None,
    };
    if let Some(role) = static_role {
        return Ok((ParameterResidency::Static, role));
    }
    let (index, suffix) = path
        .strip_prefix("layers.")
        .and_then(|p| p.split_once('.'))
        .ok_or_else(|| {
            DecoderRecipeError::invalid_config(format!(
                "unknown Qwen3.5 canonical parameter {path}"
            ))
        })?;
    let layer = index
        .parse::<usize>()
        .map_err(|_| DecoderRecipeError::invalid_config("invalid canonical layer index"))?;
    let role = match suffix {
        "input_norm.weight" => TensorRole::AttentionNorm,
        "post_attention_norm.weight" => TensorRole::FeedForwardNorm,
        "attention.query_gate.weight" => TensorRole::AttentionQuery,
        "attention.key.weight" => TensorRole::AttentionKey,
        "attention.value.weight" => TensorRole::AttentionValue,
        "attention.output.weight" | "linear_attention.output.weight" => TensorRole::AttentionOutput,
        "attention.query_norm.weight" => TensorRole::AttentionQueryNorm,
        "attention.key_norm.weight" => TensorRole::AttentionKeyNorm,
        "linear_attention.query_key_value.weight" => TensorRole::LinearAttentionQkv,
        "linear_attention.gate.weight" => TensorRole::LinearAttentionZ,
        "linear_attention.beta.weight" => TensorRole::LinearAttentionBeta,
        "linear_attention.decay.weight" => TensorRole::LinearAttentionA,
        "linear_attention.convolution.weight" => TensorRole::LinearAttentionConv,
        "linear_attention.decay_log.weight" => TensorRole::LinearAttentionALog,
        "linear_attention.time_bias.weight" => TensorRole::LinearAttentionDtBias,
        "linear_attention.norm.weight" => TensorRole::LinearAttentionNorm,
        "feed_forward.gate.weight" => TensorRole::DenseMlpGate,
        "feed_forward.up.weight" => TensorRole::DenseMlpUp,
        "feed_forward.down.weight" => TensorRole::DenseMlpDown,
        _ => {
            return Err(DecoderRecipeError::invalid_config(format!(
                "unassigned Qwen3.5 role for {path}"
            )));
        }
    };
    Ok((ParameterResidency::layer(layer), role))
}
