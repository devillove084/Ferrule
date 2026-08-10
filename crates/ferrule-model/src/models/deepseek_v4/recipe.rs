//! DeepSeek-V4 recipe for the model-independent transformer graph.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::checkpoint::HfSafetensorsInventory;
#[cfg(feature = "cuda")]
use crate::moe::streaming::ExpertMemoryPolicy;
use crate::nn::{
    DTypeConstraint, ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
    StorageEncoding,
};
use crate::support::TensorRole;
use crate::transformer::{
    Attention, DecoderAttachmentSpec, DecoderLayer, DecoderModelParts, DecoderModelSpec,
    DecoderRecipe, DecoderRecipeError, Embedding, FeedForward, Linear, MlaAttention, Moe,
    MoeRouterSpec, NameMapper, ProposalAttachmentParts, ProposalAttachmentSpec, ProposalHeadsSpec,
    Residual, RmsNorm, RotaryEmbedding, RotaryPairing, RotaryRegion, RotaryScaling,
    RouterScoreFunction, RouterSelection, SharedKvMlaDimensions, StateDictSchema,
    StateDictSchemaBuilder, SwiGlu,
};

use super::{DeepSeekV4Config, DeepSeekV4NameMapper};

const FP8_BLOCK_ROWS: usize = 128;
const FP8_BLOCK_COLS: usize = 128;
const FP4_BLOCK_SIZE: usize = 32;

/// Builds exact DeepSeek-V4 graph and state-dict metadata without changing the
/// legacy DeepSeek runtime selection path.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeepSeekV4Recipe;

impl DeepSeekV4Recipe {
    pub const fn new() -> Self {
        Self
    }

    fn parsed_config(value: &serde_json::Value) -> Result<DeepSeekV4Config, DecoderRecipeError> {
        DeepSeekV4Config::from_value(value).map_err(DecoderRecipeError::config)
    }

    fn build_spec_for(
        config: &DeepSeekV4Config,
        mtp_stages: usize,
    ) -> Result<DecoderModelSpec, DecoderRecipeError> {
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let ratio = config.compress_ratios.get(layer).copied().unwrap_or(0);
            layers.push(build_layer(config, layer, ratio, false)?);
        }
        let output_hc = config.hc_config().head_spec()?;
        let mut spec = DecoderModelSpec::new(DecoderModelParts {
            architecture: config.architecture.clone(),
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            max_sequence_length: Some(config.max_position_embeddings),
            token_embedding: Embedding::new(config.vocab_size, config.hidden_size, None)?,
            layers,
            final_norm: RmsNorm::new(config.hidden_size, config.norm_eps)?,
            output: Linear::new(config.hidden_size, config.vocab_size, false)?,
            tie_word_embeddings: false,
        })?
        .with_output_hyper_connection(output_hc)?;

        if mtp_stages != 0 {
            spec = spec.with_attachment(DecoderAttachmentSpec::Proposal(build_proposal_spec(
                config, mtp_stages,
            )?))?;
        }
        Ok(spec)
    }

    fn build_schema_for(
        config: &DeepSeekV4Config,
        mtp_stages: usize,
    ) -> Result<StateDictSchema, DecoderRecipeError> {
        let mut schema = SchemaWriter::new();
        schema.dense(
            "token_embedding.weight",
            exact(ParameterDType::Bf16),
            [config.vocab_size, config.hidden_size],
            ParameterResidency::Static,
            TensorRole::TokenEmbedding,
        )?;
        schema.dense(
            "final_norm.weight",
            float_vector_dtype()?,
            [config.hidden_size],
            ParameterResidency::Static,
            TensorRole::OutputNorm,
        )?;
        schema.dense(
            "output.weight",
            exact(ParameterDType::Bf16),
            [config.vocab_size, config.hidden_size],
            ParameterResidency::Static,
            TensorRole::OutputHead,
        )?;
        register_hyper_connection_head(
            &mut schema,
            "output_hyper_connection",
            config,
            ParameterResidency::Static,
            TensorRole::AuxOutputHiddenCompressor,
        )?;

        for layer in 0..config.num_layers {
            let ratio = config.compress_ratios.get(layer).copied().unwrap_or(0);
            register_block(
                &mut schema,
                &format!("layers.{layer}"),
                config,
                ratio,
                ParameterResidency::layer(layer),
                Some(layer),
                layer < config.num_hash_layers,
            )?;
        }

        for stage in 0..mtp_stages {
            let attachment = ModulePath::new(format!("attachments.mtp.{stage}"))?;
            let residency = ParameterResidency::Attachment { path: attachment };
            let prefix = format!("attachments.mtp.{stage}");
            register_block(
                &mut schema,
                &prefix,
                config,
                0,
                residency.clone(),
                None,
                false,
            )?;
            if stage == 0 {
                let targets = config.proposal_target_layer_ids.len();
                schema.fp8(
                    format!("{prefix}.main_projection.weight"),
                    [config.hidden_size, config.hidden_size * targets],
                    residency.clone(),
                    TensorRole::SpeculativeProjection,
                )?;
                schema.dense(
                    format!("{prefix}.main_norm.weight"),
                    float_vector_dtype()?,
                    [config.hidden_size],
                    residency.clone(),
                    TensorRole::LayerNorm,
                )?;
            }
            if stage + 1 == mtp_stages {
                register_hyper_connection_head(
                    &mut schema,
                    &format!("{prefix}.heads.hyper_connection"),
                    config,
                    residency.clone(),
                    TensorRole::AuxOutputHiddenCompressor,
                )?;
                schema.dense(
                    format!("{prefix}.heads.norm.weight"),
                    float_vector_dtype()?,
                    [config.hidden_size],
                    residency.clone(),
                    TensorRole::OutputNorm,
                )?;
                let rank = config.proposal_markov_rank.ok_or_else(|| {
                    DecoderRecipeError::invalid_config(
                        "MTP inventory requires proposal/dspark markov rank",
                    )
                })?;
                schema.fp8(
                    format!("{prefix}.heads.markov_embedding.weight"),
                    [config.vocab_size, rank],
                    residency.clone(),
                    TensorRole::SpeculativeMarkovHead,
                )?;
                schema.fp8(
                    format!("{prefix}.heads.markov_output.weight"),
                    [config.vocab_size, rank],
                    residency.clone(),
                    TensorRole::SpeculativeMarkovHead,
                )?;
                schema.fp8(
                    format!("{prefix}.heads.confidence.weight"),
                    [1, config.hidden_size + rank],
                    residency,
                    TensorRole::SpeculativeConfidenceHead,
                )?;
            }
        }
        schema.finish()
    }
}

impl DecoderRecipe for DeepSeekV4Recipe {
    fn build_spec(
        &self,
        value: &serde_json::Value,
    ) -> Result<DecoderModelSpec, DecoderRecipeError> {
        Self::build_spec_for(&Self::parsed_config(value)?, 0)
    }

    fn build_schema(
        &self,
        value: &serde_json::Value,
    ) -> Result<StateDictSchema, DecoderRecipeError> {
        Self::build_schema_for(&Self::parsed_config(value)?, 0)
    }

    fn build_name_mapper(
        &self,
        value: &serde_json::Value,
    ) -> Result<Arc<dyn NameMapper>, DecoderRecipeError> {
        let parsed = Self::parsed_config(value)?;
        Ok(Arc::new(
            DeepSeekV4NameMapper::from_config(&parsed).map_err(DecoderRecipeError::name_mapper)?,
        ))
    }

    fn build_spec_with_inventory(
        &self,
        value: &serde_json::Value,
        inventory: &HfSafetensorsInventory,
    ) -> Result<DecoderModelSpec, DecoderRecipeError> {
        let stages = discover_mtp_stage_count(inventory)?;
        Self::build_spec_for(&Self::parsed_config(value)?, stages)
    }

    fn build_schema_with_inventory(
        &self,
        value: &serde_json::Value,
        inventory: &HfSafetensorsInventory,
    ) -> Result<StateDictSchema, DecoderRecipeError> {
        let stages = discover_mtp_stage_count(inventory)?;
        Self::build_schema_for(&Self::parsed_config(value)?, stages)
    }

    fn build_name_mapper_with_inventory(
        &self,
        value: &serde_json::Value,
        inventory: &HfSafetensorsInventory,
    ) -> Result<Arc<dyn NameMapper>, DecoderRecipeError> {
        let parsed = Self::parsed_config(value)?;
        let stages = discover_mtp_stage_count(inventory)?;
        Ok(Arc::new(
            DeepSeekV4NameMapper::from_config_with_mtp_stages(&parsed, stages)
                .map_err(DecoderRecipeError::name_mapper)?,
        ))
    }
}

fn build_layer(
    config: &DeepSeekV4Config,
    index: usize,
    compress_ratio: usize,
    mtp: bool,
) -> Result<DecoderLayer, DecoderRecipeError> {
    let rotary = if compress_ratio == 0 {
        RotaryEmbedding::new(
            config.head_dim,
            config.rope_theta,
            RotaryPairing::Interleaved,
            RotaryRegion::Tail {
                dimensions: config.qk_rope_head_dim,
            },
            RotaryScaling::None,
        )?
    } else {
        RotaryEmbedding::new(
            config.head_dim,
            config.compress_rope_theta,
            RotaryPairing::Interleaved,
            RotaryRegion::Tail {
                dimensions: config.qk_rope_head_dim,
            },
            RotaryScaling::YaRN {
                factor: config.rope_factor,
                original_max_position_embeddings: config.original_seq_len,
                beta_fast: config.beta_fast as f32,
                beta_slow: config.beta_slow as f32,
                attention_factor: None,
            },
        )?
    };
    let attention = MlaAttention::shared_kv_grouped_output(
        SharedKvMlaDimensions {
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            q_lora_rank: config.q_lora_rank,
            rope_head_dim: config.qk_rope_head_dim,
            output_groups: config.o_groups,
            output_rank: config.o_lora_rank,
            window_size: config.window_size,
            compress_ratio,
            index_heads: config.index_n_heads,
            index_head_dim: config.index_head_dim,
            index_topk: config.index_topk,
        },
        config.norm_eps,
        false,
        rotary,
    )?;
    let hash = !mtp && index < config.num_hash_layers;
    let selection = if hash {
        RouterSelection::HashAssistedTopK {
            hash_layers: config.num_hash_layers,
        }
    } else {
        RouterSelection::TopK
    };
    let router = MoeRouterSpec::new(
        config.num_routed_experts,
        config.num_experts_per_tok,
        RouterScoreFunction::SqrtSoftplus,
        selection,
        true,
        config.route_scale,
    )?
    .with_selection_bias(!hash);
    let shared = SwiGlu::new(config.hidden_size, config.moe_intermediate_size, false)?
        .with_activation_limit(config.swiglu_limit)?;
    let moe = Moe::new(
        config.hidden_size,
        config.moe_intermediate_size,
        router,
        false,
    )?
    .with_activation_limit(config.swiglu_limit)?
    .with_shared_expert(shared)?;
    let hyper = Residual::Hyper(config.hc_config().residual_spec()?);
    Ok(DecoderLayer::new(
        index,
        RmsNorm::new(config.hidden_size, config.norm_eps)?,
        Attention::Mla(attention),
        hyper.clone(),
        RmsNorm::new(config.hidden_size, config.norm_eps)?,
        FeedForward::Moe(moe),
        hyper,
    )?)
}

fn build_proposal_spec(
    config: &DeepSeekV4Config,
    stages: usize,
) -> Result<ProposalAttachmentSpec, DecoderRecipeError> {
    if config.proposal_target_layer_ids.is_empty() {
        return Err(DecoderRecipeError::invalid_config(
            "MTP inventory requires proposal/dspark target layer IDs",
        ));
    }
    let markov_rank = config.proposal_markov_rank.ok_or_else(|| {
        DecoderRecipeError::invalid_config("MTP inventory requires proposal/dspark markov rank")
    })?;
    let mut stage_specs = Vec::with_capacity(stages);
    for stage in 0..stages {
        stage_specs.push(build_layer(config, stage, 0, true)?);
    }
    let heads = ProposalHeadsSpec::new(
        config.hc_config().head_spec()?,
        RmsNorm::new(config.hidden_size, config.norm_eps)?,
        config.vocab_size,
        markov_rank,
        config.hidden_size,
    )?;
    Ok(ProposalAttachmentSpec::new(ProposalAttachmentParts {
        path: "attachments.mtp".into(),
        hidden_size: config.hidden_size,
        block_size: config.proposal_block_size,
        noise_token_id: config.proposal_noise_token_id,
        target_layer_ids: config.proposal_target_layer_ids.clone(),
        stages: stage_specs,
        main_projection: Some(Linear::new(
            config
                .hidden_size
                .checked_mul(config.proposal_target_layer_ids.len())
                .ok_or_else(|| {
                    DecoderRecipeError::invalid_config(
                        "proposal main projection input width overflows usize",
                    )
                })?,
            config.hidden_size,
            false,
        )?),
        main_norm: Some(RmsNorm::new(config.hidden_size, config.norm_eps)?),
        heads: Some(heads),
    })?)
}

fn discover_mtp_stage_count(
    inventory: &HfSafetensorsInventory,
) -> Result<usize, DecoderRecipeError> {
    let mut stages = BTreeSet::new();
    for tensor in &inventory.tensors {
        let Some(rest) = tensor.name.strip_prefix("mtp.") else {
            continue;
        };
        let stage = rest.split('.').next().unwrap_or_default();
        if stage.is_empty() || !stage.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(DecoderRecipeError::invalid_config(format!(
                "malformed MTP tensor name '{}'",
                tensor.name
            )));
        }
        let stage = stage.parse::<usize>().map_err(|_| {
            DecoderRecipeError::invalid_config(format!(
                "MTP stage index overflows usize in '{}'",
                tensor.name
            ))
        })?;
        stages.insert(stage);
    }
    for (expected, actual) in stages.iter().copied().enumerate() {
        if actual != expected {
            return Err(DecoderRecipeError::invalid_config(format!(
                "MTP stages must be contiguous from zero; stage {expected} is missing"
            )));
        }
    }
    Ok(stages.len())
}

fn register_block(
    schema: &mut SchemaWriter,
    prefix: &str,
    config: &DeepSeekV4Config,
    compress_ratio: usize,
    residency: ParameterResidency,
    layer: Option<usize>,
    hash_router: bool,
) -> Result<(), DecoderRecipeError> {
    schema.dense(
        format!("{prefix}.input_norm.weight"),
        float_vector_dtype()?,
        [config.hidden_size],
        residency.clone(),
        TensorRole::AttentionNorm,
    )?;
    register_attention(schema, prefix, config, compress_ratio, residency.clone())?;
    register_hyper_connection_layer(
        schema,
        &format!("{prefix}.hyper_connection.attention"),
        config,
        residency.clone(),
    )?;
    schema.dense(
        format!("{prefix}.post_attention_norm.weight"),
        float_vector_dtype()?,
        [config.hidden_size],
        residency.clone(),
        TensorRole::FeedForwardNorm,
    )?;
    register_moe(
        schema,
        prefix,
        config,
        residency.clone(),
        layer,
        hash_router,
    )?;
    register_hyper_connection_layer(
        schema,
        &format!("{prefix}.hyper_connection.feed_forward"),
        config,
        residency,
    )
}

fn register_attention(
    schema: &mut SchemaWriter,
    prefix: &str,
    config: &DeepSeekV4Config,
    compress_ratio: usize,
    residency: ParameterResidency,
) -> Result<(), DecoderRecipeError> {
    let attention = format!("{prefix}.attention");
    let query_width = config.num_heads * config.head_dim;
    let output_latent = config.o_groups * config.o_lora_rank;
    for (name, shape, role) in [
        (
            "query_a",
            [config.q_lora_rank, config.hidden_size],
            TensorRole::AttentionLatentQueryA,
        ),
        (
            "query_b",
            [query_width, config.q_lora_rank],
            TensorRole::AttentionLatentQueryB,
        ),
        (
            "key_value",
            [config.head_dim, config.hidden_size],
            TensorRole::AttentionLatentKv,
        ),
        (
            "output_a",
            [output_latent, query_width / config.o_groups],
            TensorRole::AttentionLatentOutputA,
        ),
        (
            "output_b",
            [config.hidden_size, output_latent],
            TensorRole::AttentionLatentOutputB,
        ),
    ] {
        schema.fp8(
            format!("{attention}.{name}.weight"),
            shape,
            residency.clone(),
            role,
        )?;
    }
    schema.dense(
        format!("{attention}.query_norm.weight"),
        float_vector_dtype()?,
        [config.q_lora_rank],
        residency.clone(),
        TensorRole::AttentionQueryNorm,
    )?;
    schema.dense(
        format!("{attention}.key_value_norm.weight"),
        float_vector_dtype()?,
        [config.head_dim],
        residency.clone(),
        TensorRole::AttentionKeyValueNorm,
    )?;
    schema.dense(
        format!("{attention}.sink"),
        exact(ParameterDType::F32),
        [config.num_heads],
        residency.clone(),
        TensorRole::AttentionSink,
    )?;
    if compress_ratio != 0 {
        register_compressor(
            schema,
            &format!("{attention}.compressor"),
            config.hidden_size,
            config.head_dim,
            compress_ratio,
            residency.clone(),
            TensorRole::AttentionCompressor,
        )?;
    }
    if compress_ratio == 4 {
        register_compressor(
            schema,
            &format!("{attention}.indexer.compressor"),
            config.hidden_size,
            config.index_head_dim,
            compress_ratio,
            residency.clone(),
            TensorRole::AuxIndexer,
        )?;
        schema.fp8(
            format!("{attention}.indexer.query.weight"),
            [
                config.index_n_heads * config.index_head_dim,
                config.q_lora_rank,
            ],
            residency.clone(),
            TensorRole::AuxIndexer,
        )?;
        schema.dense(
            format!("{attention}.indexer.weights.weight"),
            exact(ParameterDType::Bf16),
            [config.index_n_heads, config.hidden_size],
            residency,
            TensorRole::AuxIndexer,
        )?;
    }
    Ok(())
}

fn register_compressor(
    schema: &mut SchemaWriter,
    prefix: &str,
    hidden_size: usize,
    head_dim: usize,
    ratio: usize,
    residency: ParameterResidency,
    role: TensorRole,
) -> Result<(), DecoderRecipeError> {
    let coefficient = if ratio == 4 { 2 } else { 1 };
    let output = coefficient * head_dim;
    schema.dense(
        format!("{prefix}.ape"),
        exact(ParameterDType::F32),
        [ratio, output],
        residency.clone(),
        role.clone(),
    )?;
    schema.dense(
        format!("{prefix}.norm.weight"),
        float_vector_dtype()?,
        [head_dim],
        residency.clone(),
        role.clone(),
    )?;
    schema.dense(
        format!("{prefix}.key_value.weight"),
        dense_linear_dtype()?,
        [output, hidden_size],
        residency.clone(),
        role.clone(),
    )?;
    schema.dense(
        format!("{prefix}.gate.weight"),
        dense_linear_dtype()?,
        [output, hidden_size],
        residency,
        role,
    )?;
    Ok(())
}

fn register_moe(
    schema: &mut SchemaWriter,
    prefix: &str,
    config: &DeepSeekV4Config,
    residency: ParameterResidency,
    layer: Option<usize>,
    hash_router: bool,
) -> Result<(), DecoderRecipeError> {
    schema.dense(
        format!("{prefix}.router.weight"),
        exact(ParameterDType::Bf16),
        [config.num_routed_experts, config.hidden_size],
        residency.clone(),
        TensorRole::RouterLogits,
    )?;
    if hash_router {
        schema.dense(
            format!("{prefix}.router.hash"),
            DTypeConstraint::one_of([ParameterDType::I32, ParameterDType::I64])?,
            [config.vocab_size, config.num_experts_per_tok],
            residency.clone(),
            TensorRole::HashRouterTable,
        )?;
    } else {
        schema.dense(
            format!("{prefix}.router.bias"),
            exact(ParameterDType::F32),
            [config.num_routed_experts],
            residency.clone(),
            TensorRole::RouterBias,
        )?;
    }
    for (name, shape, role) in expert_matrices(config) {
        schema.fp8(
            format!("{prefix}.shared_expert.{name}.weight"),
            shape,
            residency.clone(),
            shared_role(role),
        )?;
    }
    for expert in 0..config.num_routed_experts {
        let expert_residency = layer.map_or_else(
            || residency.clone(),
            |layer| ParameterResidency::expert(layer, expert),
        );
        for (name, shape, role) in expert_matrices(config) {
            schema.fp4(
                format!("{prefix}.experts.{expert}.{name}.weight"),
                shape,
                expert_residency.clone(),
                role,
            )?;
        }
    }
    Ok(())
}

fn expert_matrices(config: &DeepSeekV4Config) -> [(&'static str, [usize; 2], TensorRole); 3] {
    [
        (
            "gate",
            [config.moe_intermediate_size, config.hidden_size],
            TensorRole::RoutedExpertGate,
        ),
        (
            "up",
            [config.moe_intermediate_size, config.hidden_size],
            TensorRole::RoutedExpertUp,
        ),
        (
            "down",
            [config.hidden_size, config.moe_intermediate_size],
            TensorRole::RoutedExpertDown,
        ),
    ]
}

fn shared_role(role: TensorRole) -> TensorRole {
    match role {
        TensorRole::RoutedExpertGate => TensorRole::SharedExpertGate,
        TensorRole::RoutedExpertUp => TensorRole::SharedExpertUp,
        TensorRole::RoutedExpertDown => TensorRole::SharedExpertDown,
        _ => unreachable!("expert matrix helper only returns routed expert roles"),
    }
}

fn register_hyper_connection_layer(
    schema: &mut SchemaWriter,
    prefix: &str,
    config: &DeepSeekV4Config,
    residency: ParameterResidency,
) -> Result<(), DecoderRecipeError> {
    let hyper_connection = config.hc_config();
    schema.dense(
        format!("{prefix}.function"),
        exact(ParameterDType::F32),
        hyper_connection.function_shape(),
        residency.clone(),
        TensorRole::AuxHiddenCompressor,
    )?;
    schema.dense(
        format!("{prefix}.scale"),
        exact(ParameterDType::F32),
        hyper_connection.scale_shape(),
        residency.clone(),
        TensorRole::AuxHiddenCompressor,
    )?;
    schema.dense(
        format!("{prefix}.base"),
        exact(ParameterDType::F32),
        hyper_connection.base_shape(),
        residency,
        TensorRole::AuxHiddenCompressor,
    )?;
    Ok(())
}

fn register_hyper_connection_head(
    schema: &mut SchemaWriter,
    prefix: &str,
    config: &DeepSeekV4Config,
    residency: ParameterResidency,
    role: TensorRole,
) -> Result<(), DecoderRecipeError> {
    let hyper_connection = config.hc_config();
    schema.dense(
        format!("{prefix}.function"),
        exact(ParameterDType::F32),
        hyper_connection.head_function_shape(),
        residency.clone(),
        role.clone(),
    )?;
    schema.dense(
        format!("{prefix}.scale"),
        exact(ParameterDType::F32),
        hyper_connection.head_scale_shape(),
        residency.clone(),
        role.clone(),
    )?;
    schema.dense(
        format!("{prefix}.base"),
        exact(ParameterDType::F32),
        hyper_connection.head_base_shape(),
        residency,
        role,
    )?;
    Ok(())
}

struct SchemaWriter {
    builder: StateDictSchemaBuilder,
    next_id: u64,
}

impl SchemaWriter {
    fn new() -> Self {
        Self {
            builder: StateDictSchema::builder(),
            next_id: 1,
        }
    }

    fn dense(
        &mut self,
        path: impl Into<String>,
        dtype: DTypeConstraint,
        shape: impl Into<Vec<usize>>,
        residency: ParameterResidency,
        role: TensorRole,
    ) -> Result<ParameterId, DecoderRecipeError> {
        let id = self.id()?;
        let parameter =
            ParameterSpec::new(id, ModulePath::new(path.into())?, dtype, shape, residency)?;
        self.builder.register_with_role(parameter, role)?;
        Ok(id)
    }

    fn fp8(
        &mut self,
        path: impl Into<String>,
        shape: [usize; 2],
        residency: ParameterResidency,
        role: TensorRole,
    ) -> Result<ParameterId, DecoderRecipeError> {
        let id = self.id()?;
        let parameter = ParameterSpec::new_encoded(
            id,
            ModulePath::new(path.into())?,
            ParameterDType::F8E4M3,
            shape,
            shape,
            StorageEncoding::Fp8Block128,
            residency,
        )?
        .with_required_scale(
            ParameterDType::F8E8M0,
            [
                shape[0].div_ceil(FP8_BLOCK_ROWS),
                shape[1].div_ceil(FP8_BLOCK_COLS),
            ],
        )?;
        self.builder.register_with_role(parameter, role)?;
        Ok(id)
    }

    fn fp4(
        &mut self,
        path: impl Into<String>,
        shape: [usize; 2],
        residency: ParameterResidency,
        role: TensorRole,
    ) -> Result<ParameterId, DecoderRecipeError> {
        let id = self.id()?;
        let parameter = ParameterSpec::new_encoded(
            id,
            ModulePath::new(path.into())?,
            ParameterDType::I8,
            shape,
            [shape[0], shape[1] / 2],
            StorageEncoding::PackedFp4X2 {
                block_size: FP4_BLOCK_SIZE,
            },
            residency,
        )?
        .with_required_scale(
            ParameterDType::F8E8M0,
            [shape[0], shape[1] / FP4_BLOCK_SIZE],
        )?;
        self.builder.register_with_role(parameter, role)?;
        Ok(id)
    }

    fn id(&mut self) -> Result<ParameterId, DecoderRecipeError> {
        let id = ParameterId::new(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| {
            DecoderRecipeError::invalid_config("DeepSeek-V4 parameter ID overflow")
        })?;
        Ok(id)
    }

    fn finish(self) -> Result<StateDictSchema, DecoderRecipeError> {
        Ok(self.builder.build()?)
    }
}

fn exact(dtype: ParameterDType) -> DTypeConstraint {
    DTypeConstraint::exact(dtype)
}

fn float_vector_dtype() -> Result<DTypeConstraint, DecoderRecipeError> {
    Ok(exact(ParameterDType::Bf16))
}

fn dense_linear_dtype() -> Result<DTypeConstraint, DecoderRecipeError> {
    Ok(DTypeConstraint::one_of([
        ParameterDType::Bf16,
        ParameterDType::F32,
    ])?)
}

// ---------------------------------------------------------------------------
// Prepared family assembly: bound state dict -> MLA/HC/routed-MoE/MTP
// components, prepared model plan, and family execution policy.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
use super::checkpoint::DeepSeekV4Checkpoint;
#[cfg(feature = "cuda")]
use super::config::{
    with_deepseek_v4_attention_execution_policies, with_deepseek_v4_linear_execution_policy,
    with_deepseek_v4_swiglu_execution_policies,
};
#[cfg(feature = "cuda")]
use crate::checkpoint::weight::LinearWeight;
#[cfg(feature = "cuda")]
use crate::execution::TransformerStage;
#[cfg(feature = "cuda")]
use crate::ffn::SwiGluFfnPayload;
#[cfg(feature = "cuda")]
use crate::materialization::MaterializationSourceCatalog;
#[cfg(feature = "cuda")]
use crate::moe::routing::{ExpertRouterPolicy, RouterWeights};
#[cfg(feature = "cuda")]
use crate::moe::streaming::{ExpertLoadSource, ExpertSourceCatalog};
#[cfg(feature = "cuda")]
use crate::moe::{ExpertLayerSources, RoutedMoePayload};
#[cfg(feature = "cuda")]
use crate::transformer::attention::mla::{
    MlaCompression, MlaCompressor, MlaConfig, MlaIndexer, MlaKvLayout, MlaWeights, PreparedMla,
};
#[cfg(feature = "cuda")]
use crate::transformer::connection::{
    HyperConnection, HyperConnectionHead, HyperConnectionPhase, HyperConnectionWeights,
};
#[cfg(feature = "cuda")]
use crate::transformer::cuda::{
    CudaTransformerMtpSource, CudaTransformerSource, MlaHyperMoeLayer, PreparedCudaTransformer,
};
#[cfg(feature = "cuda")]
use crate::transformer::proposal::{MtpAttachment, MtpHeads, MtpStage};
#[cfg(feature = "cuda")]
use crate::transformer::{
    BoundMatrix, BoundParameter, PreparedDecoderAttachment, PreparedNorm, StateDictMaterializer,
};
#[cfg(feature = "cuda")]
use ferrule_backend::plan::ModelKernelPlan;
#[cfg(feature = "cuda")]
use ferrule_common::Error;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4OutputProfileStats {
    pub packed_prefill_batches: u64,
    pub packed_prefill_rows: u64,
    pub packed_decode_batches: u64,
    pub packed_decode_rows: u64,
    pub packed_mixed_batches: u64,
    pub packed_mixed_rows: u64,
    pub final_hc_head_calls: u64,
    pub final_hc_head_us: u64,
    pub final_norm_calls: u64,
    pub final_norm_us: u64,
    pub lm_head_topk_calls: u64,
    pub lm_head_topk_us: u64,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4PrepareProfile {
    pub validation_us: u64,
    pub attachment_bind_us: u64,
    pub target_bind_us: u64,
    pub execution_plan_us: u64,
    pub manifest_us: u64,
    pub total_us: u64,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4PrepareOptions {
    pub max_layers: usize,
    pub output_head_chunk_rows: usize,
    pub expert_reader_max_tensor_bytes: u64,
    pub expert_memory_policy: ExpertMemoryPolicy,
    pub moe_hotset_experts: usize,
    pub reserved_device_bytes: u64,
}

#[cfg(feature = "cuda")]
impl Default for DeepSeekV4PrepareOptions {
    fn default() -> Self {
        Self {
            max_layers: crate::families::deepseek_v4::NUM_LAYERS,
            output_head_chunk_rows: 1024,
            expert_reader_max_tensor_bytes: 64 * 1024 * 1024,
            expert_memory_policy: ExpertMemoryPolicy::default(),
            moe_hotset_experts: 0,
            reserved_device_bytes: 8 * 1024 * 1024 * 1024,
        }
    }
}

/// CUDA family assembly: prepared resources, execution policy, and the
/// materialized MLA/HC/routed-MoE/MTP component constructors.
#[cfg(feature = "cuda")]
mod assembly {
    use super::*;
    use ferrule_common::Result;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct DeepSeekV4ExecutionPolicy {
        prefill_progress: bool,
        managed_experts: bool,
        expert_upload_inflight: usize,
        profile: bool,
        profile_sync: bool,
    }

    #[cfg(feature = "cuda")]
    impl Default for DeepSeekV4ExecutionPolicy {
        fn default() -> Self {
            Self {
                prefill_progress: false,
                managed_experts: true,
                expert_upload_inflight: 32,
                profile: false,
                profile_sync: false,
            }
        }
    }

    #[cfg(feature = "cuda")]
    impl DeepSeekV4ExecutionPolicy {
        pub const fn prefill_progress(&self) -> bool {
            self.prefill_progress
        }
        pub const fn managed_experts(&self) -> bool {
            self.managed_experts
        }
        pub const fn expert_upload_inflight(&self) -> usize {
            self.expert_upload_inflight
        }
        pub const fn profile_enabled(&self) -> bool {
            self.profile
        }
        pub const fn profile_sync(&self) -> bool {
            self.profile_sync
        }
        pub(crate) fn resolve() -> Result<Self> {
            let get = |name| std::env::var(name).ok();
            let prefill_progress = env_bool(
                "FERRULE_DSV4_PREFILL_PROGRESS",
                get("FERRULE_DSV4_PREFILL_PROGRESS"),
                false,
            )?;
            let managed_experts = env_bool(
                "FERRULE_MANAGED_EXPERTS",
                get("FERRULE_MANAGED_EXPERTS"),
                true,
            )?;
            let expert_upload_inflight = env_usize(
                "FERRULE_DSV4_EXPERT_UPLOAD_INFLIGHT",
                get("FERRULE_DSV4_EXPERT_UPLOAD_INFLIGHT"),
                32,
            )?;
            let profile_sync = env_bool(
                "FERRULE_DSV4_PROFILE_SYNC",
                get("FERRULE_DSV4_PROFILE_SYNC"),
                false,
            )?;
            let profile = profile_sync
                || prefill_progress
                || env_bool("FERRULE_DSV4_PROFILE", get("FERRULE_DSV4_PROFILE"), false)?;
            Ok(Self {
                prefill_progress,
                managed_experts,
                expert_upload_inflight,
                profile,
                profile_sync,
            })
        }
    }

    #[cfg(feature = "cuda")]
    fn env_bool(name: &str, value: Option<String>, default: bool) -> Result<bool> {
        match value.as_deref() {
            None => Ok(default),
            Some("1" | "true" | "yes" | "on") => Ok(true),
            Some("0" | "false" | "no" | "off") => Ok(false),
            Some(value) => Err(prepare_error(format!(
                "environment variable {name} has invalid boolean '{value}'"
            ))),
        }
    }

    #[cfg(feature = "cuda")]
    fn env_usize(name: &str, value: Option<String>, default: usize) -> Result<usize> {
        let Some(value) = value else {
            return Ok(default);
        };
        value
            .parse()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                prepare_error(format!(
                    "environment variable {name} has invalid usize '{value}'"
                ))
            })
    }

    #[cfg(feature = "cuda")]
    fn prepare_error(message: impl Into<String>) -> Error {
        Error::Model {
            message: format!("DeepSeek-V4 prepare: {}", message.into()),
        }
    }

    #[cfg(feature = "cuda")]
    fn path(value: &str) -> Result<ModulePath> {
        ModulePath::new(value).map_err(|error| prepare_error(error.to_string()))
    }

    #[cfg(feature = "cuda")]
    fn deepseek_linear(
        materializer: &StateDictMaterializer,
        binding: &BoundParameter,
    ) -> Result<LinearWeight> {
        Ok(with_deepseek_v4_linear_execution_policy(
            materializer.linear(binding)?,
        ))
    }

    /// Static (non-layer) DeepSeek weights bound through the generic state dict.
    #[cfg(feature = "cuda")]
    #[derive(Debug, Clone)]
    pub(crate) struct DeepSeekV4StaticResources {
        embedding: BoundMatrix,
        output_norm: Arc<[f32]>,
        output_head: BoundMatrix,
        hyper_connection_head: HyperConnectionHead,
    }

    #[cfg(feature = "cuda")]
    impl DeepSeekV4StaticResources {
        pub const fn embedding(&self) -> &BoundMatrix {
            &self.embedding
        }
        pub fn output_norm(&self) -> &[f32] {
            &self.output_norm
        }
        pub const fn output_head(&self) -> &BoundMatrix {
            &self.output_head
        }
        pub const fn hyper_connection_head(&self) -> &HyperConnectionHead {
            &self.hyper_connection_head
        }
    }

    /// Fully assembled DeepSeek-V4 family resources: composed generic components,
    /// expert source catalogs, KV layout, kernel plans, and execution policy.
    pub struct DeepSeekV4Resources {
        descriptor: crate::ModelDescriptor,
        prepared: PreparedDecoderAttachment<PreparedCudaTransformer>,
        config: DeepSeekV4Config,
        options: DeepSeekV4PrepareOptions,
        statics: DeepSeekV4StaticResources,
        layers: Box<[MlaHyperMoeLayer]>,
        layer_experts: Box<[ExpertLayerSources]>,
        proposal_attachment: Option<MtpAttachment<MlaHyperMoeLayer, Arc<ExpertSourceCatalog>>>,
        proposal_stage_experts: Box<[ExpertLayerSources]>,
        expert_materialization_sources: Arc<MaterializationSourceCatalog<ExpertLoadSource>>,
        kv_layout: MlaKvLayout,
        policy: DeepSeekV4ExecutionPolicy,
        kernel_plan: ModelKernelPlan,
        mtp_transformer_kernel_plan: Option<ModelKernelPlan>,
    }

    impl DeepSeekV4Resources {
        pub const fn descriptor(&self) -> &crate::ModelDescriptor {
            &self.descriptor
        }
        pub(crate) const fn prepared(&self) -> &PreparedDecoderAttachment<PreparedCudaTransformer> {
            &self.prepared
        }
        pub(crate) const fn config(&self) -> &DeepSeekV4Config {
            &self.config
        }
        pub fn model_info(&self) -> crate::runner::ModelInfo {
            crate::runner::ModelInfo {
                family: self.descriptor.spec.family.clone(),
                architecture: self.descriptor.spec.architecture.clone(),
                attention: self.descriptor.spec.attention.clone(),
                weight_source: self.descriptor.spec.weight_source,
                hidden_size: self.config.hidden_size,
                num_layers: self.config.num_layers,
                num_experts: self.config.num_routed_experts,
                num_experts_per_tok: self.config.num_experts_per_tok,
                vocab_size: self.config.vocab_size,
                backend: "cuda",
            }
        }
        pub const fn prepare_options(&self) -> &DeepSeekV4PrepareOptions {
            &self.options
        }
        pub fn layers(&self) -> &[MlaHyperMoeLayer] {
            &self.layers
        }
        pub fn layer_experts(&self) -> &[ExpertLayerSources] {
            &self.layer_experts
        }
        pub const fn proposal_attachment(
            &self,
        ) -> Option<&MtpAttachment<MlaHyperMoeLayer, Arc<ExpertSourceCatalog>>> {
            self.proposal_attachment.as_ref()
        }
        pub fn proposal_stage_experts(&self) -> &[ExpertLayerSources] {
            &self.proposal_stage_experts
        }
        pub const fn kv_layout(&self) -> &MlaKvLayout {
            &self.kv_layout
        }
        pub const fn policy(&self) -> &DeepSeekV4ExecutionPolicy {
            &self.policy
        }
        pub(crate) const fn expert_materialization_sources(
            &self,
        ) -> &Arc<MaterializationSourceCatalog<ExpertLoadSource>> {
            &self.expert_materialization_sources
        }

        /// Typed CUDA compilation view over the assembled components.
        pub(crate) fn cuda_source(
            &self,
        ) -> Result<CudaTransformerSource<'_, Arc<ExpertSourceCatalog>>> {
            let mtp = self
                .proposal_attachment
                .as_ref()
                .map(|attachment| {
                    let kernel_plan = self
                        .mtp_transformer_kernel_plan
                        .as_ref()
                        .ok_or_else(|| prepare_error("MTP transformer kernel plan is missing"))?;
                    Ok::<_, Error>(CudaTransformerMtpSource {
                        attachment,
                        kernel_plan,
                    })
                })
                .transpose()?;
            Ok(CudaTransformerSource {
                embedding: self.statics.embedding().matrix(),
                output_norm: self.statics.output_norm(),
                output_head: self.statics.output_head().matrix(),
                output_hyper_connection_head: self.statics.hyper_connection_head(),
                layers: &self.layers,
                kernel_plan: &self.kernel_plan,
                mtp,
            })
        }
    }

    /// Immutable generation-stamped DeepSeek model plan.
    #[cfg(feature = "cuda")]
    pub type DeepSeekV4ModelPlan = crate::transformer::PreparedDecoder<
        Arc<DeepSeekV4Resources>,
        TransformerStage,
        PreparedCudaTransformer,
    >;

    #[cfg(feature = "cuda")]
    pub fn prepare(
        model: &DeepSeekV4Checkpoint,
        options: DeepSeekV4PrepareOptions,
    ) -> Result<DeepSeekV4ModelPlan> {
        let policy = DeepSeekV4ExecutionPolicy::resolve()?;
        prepare_with_policy(model, options, policy)
    }

    // The plan Arc is dictated by the generic `PreparedDecoder` container; the
    // assembled resources hold single-threaded CUDA handles and never cross
    // threads because the resident runner owns them on one worker.
    #[cfg(feature = "cuda")]
    #[allow(clippy::arc_with_non_send_sync)]
    pub(crate) fn prepare_with_policy(
        model: &DeepSeekV4Checkpoint,
        options: DeepSeekV4PrepareOptions,
        policy: DeepSeekV4ExecutionPolicy,
    ) -> Result<DeepSeekV4ModelPlan> {
        validate_options(model, options)?;
        let materializer = StateDictMaterializer::new(model.max_parameter_bytes())?;
        let statics = materialize_statics(model, &materializer)?;
        let streaming = model.resolved_expert_streaming_policy(options.moe_hotset_experts);

        let mut layers = Vec::with_capacity(options.max_layers);
        let mut layer_experts = Vec::with_capacity(options.max_layers);
        for layer in 0..options.max_layers {
            layers.push(materialize_layer(
                model,
                &materializer,
                layer,
                &format!("layers.{layer}"),
                false,
            )?);
            layer_experts.push(ExpertLayerSources::new(
                model
                    .resources()
                    .decoder()
                    .source_catalogs()
                    .experts_for_layer(layer)?,
                streaming.clone(),
            ));
        }

        let proposal_attachment = (options.max_layers == model.config.num_layers)
            .then(|| materialize_proposal(model, &materializer))
            .transpose()?
            .flatten();
        let proposal_stage_experts = proposal_attachment
            .as_ref()
            .map(|attachment| {
                attachment
                    .stages
                    .iter()
                    .map(|stage| {
                        ExpertLayerSources::new(Arc::clone(&stage.experts), streaming.clone())
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
            .into_boxed_slice();

        let (kernel_plan, mtp_transformer_kernel_plan) =
            crate::transformer::cuda::compile_mla_hyper_moe_plans(
                &layers,
                proposal_attachment.as_ref(),
            )?;
        let kv_layout = build_kv_layout(model, options.max_layers, proposal_attachment.as_ref())?;

        let expert_materialization_sources = Arc::clone(
            model
                .resources()
                .decoder()
                .source_catalogs()
                .expert_materialization(),
        );

        let executable = model
            .resources()
            .decoder()
            .prepared_executable(options.max_layers)?;
        crate::transformer::PreparedDecoder::publish_with_attachment(executable, move |prepared| {
            Ok(Arc::new(DeepSeekV4Resources {
                descriptor: model.descriptor.clone(),
                prepared,
                config: model.config.clone(),
                options,
                statics,
                layers: layers.into_boxed_slice(),
                layer_experts: layer_experts.into_boxed_slice(),
                proposal_attachment,
                proposal_stage_experts,
                expert_materialization_sources,
                kv_layout,
                policy,
                kernel_plan,
                mtp_transformer_kernel_plan,
            }))
        })
    }

    #[cfg(feature = "cuda")]
    fn materialize_statics(
        model: &DeepSeekV4Checkpoint,
        materializer: &StateDictMaterializer,
    ) -> Result<DeepSeekV4StaticResources> {
        let resources = model.resources();
        let embedding = BoundMatrix::new(
            resources
                .decoder()
                .require_static(TensorRole::TokenEmbedding)?
                .clone(),
            "token embedding",
        )?;
        let output_head = BoundMatrix::new(
            resources
                .decoder()
                .require_static(TensorRole::OutputHead)?
                .clone(),
            "output head",
        )?;
        let output_norm_binding = resources
            .decoder()
            .require_static(TensorRole::OutputNorm)?
            .clone();
        let hc = [
            resources
                .decoder()
                .require_static_path(&path("output_hyper_connection.function")?)?
                .clone(),
            resources
                .decoder()
                .require_static_path(&path("output_hyper_connection.scale")?)?
                .clone(),
            resources
                .decoder()
                .require_static_path(&path("output_hyper_connection.base")?)?
                .clone(),
        ];
        Ok(DeepSeekV4StaticResources {
            embedding,
            output_norm: materializer
                .parameter(&output_norm_binding)?
                .values_f32()?
                .into(),
            output_head,
            hyper_connection_head: HyperConnectionHead::materialize(
                materializer,
                [&hc[0], &hc[1], &hc[2]],
                model.config.hc_config(),
            )?,
        })
    }

    #[cfg(feature = "cuda")]
    fn materialize_layer(
        model: &DeepSeekV4Checkpoint,
        materializer: &StateDictMaterializer,
        execution_layer: usize,
        prefix: &str,
        attachment: bool,
    ) -> Result<MlaHyperMoeLayer> {
        let attachment_path = attachment.then(|| path(prefix)).transpose()?;
        let binding = |suffix: &str| -> Result<BoundParameter> {
            let canonical = path(&format!("{prefix}.{suffix}"))?;
            match attachment_path.as_ref() {
                Some(attachment) => model
                    .resources()
                    .decoder()
                    .require_attachment_path(attachment, &canonical)
                    .cloned(),
                None => model
                    .resources()
                    .decoder()
                    .require_layer_path(execution_layer, &canonical)
                    .cloned(),
            }
        };
        let config = if attachment {
            model
                .config
                .attention_config_for_proposal_stage(execution_layer - model.config.num_layers)?
        } else {
            model.config.attention_config_for_layer(execution_layer)?
        };
        let linear = |suffix: &str| deepseek_linear(materializer, &binding(suffix)?);
        let payload = with_deepseek_v4_attention_execution_policies(MlaWeights {
            layer: execution_layer,
            query_a: linear("attention.query_a.weight")?,
            query_b: linear("attention.query_b.weight")?,
            key_value: linear("attention.key_value.weight")?,
            output_a: linear("attention.output_a.weight")?,
            output_b: linear("attention.output_b.weight")?,
            query_norm: norm_values(
                materializer,
                &binding("attention.query_norm.weight")?,
                config.norm_eps,
            )?,
            key_value_norm: norm_values(
                materializer,
                &binding("attention.key_value_norm.weight")?,
                config.norm_eps,
            )?,
            attention_sink: materializer
                .parameter(&binding("attention.sink")?)?
                .values_f32()?,
        });
        let shape_context = format!("DeepSeek-V4 layer {execution_layer}");
        crate::models::common::shape::check_linear(
            &payload.key_value,
            config.head_dim,
            config.hidden_size,
            "attention.key_value",
            &shape_context,
        )?;
        crate::models::common::shape::check_len(
            payload.key_value_norm.len(),
            config.head_dim,
            "attention.key_value_norm",
            &shape_context,
        )?;
        let attention = PreparedMla::new_with_compressed(
            execution_layer,
            config,
            payload,
            materialize_compressed(materializer, &binding, config)?,
        )?;
        let hyper_connection_config = model.config.hc_config();
        let hyper_connection_phase = |stage, norm| -> Result<HyperConnectionPhase> {
            let values = [
                binding(&format!("hyper_connection.{stage}.function"))?,
                binding(&format!("hyper_connection.{stage}.scale"))?,
                binding(&format!("hyper_connection.{stage}.base"))?,
            ];
            Ok(HyperConnectionPhase::new(
                HyperConnectionWeights::materialize(
                    materializer,
                    [&values[0], &values[1], &values[2]],
                    hyper_connection_config,
                )?,
                norm_values(materializer, &binding(norm)?, model.config.norm_eps)?,
            ))
        };
        let hyper_connection = HyperConnection::new(
            hyper_connection_config,
            hyper_connection_phase("attention", "input_norm.weight")?,
            hyper_connection_phase("feed_forward", "post_attention_norm.weight")?,
        )?;
        let hash = binding("router.hash")
            .ok()
            .map(|value| materializer.integer(&value))
            .transpose()?;
        let bias = binding("router.bias")
            .ok()
            .map(|value| materializer.parameter(&value)?.values_f32())
            .transpose()?;
        let (hash_table, hash_rows, hash_cols) = hash.map_or((None, 0, 0), |value| {
            let shape = value.shape();
            (
                Some(value.values().to_vec()),
                shape[0],
                shape.get(1).copied().unwrap_or(1),
            )
        });
        let shared_ffn = with_deepseek_v4_swiglu_execution_policies(SwiGluFfnPayload {
            gate: linear("shared_expert.gate.weight")?,
            up: linear("shared_expert.up.weight")?,
            down: linear("shared_expert.down.weight")?,
            swiglu_limit: model.config.swiglu_limit,
        });
        Ok(MlaHyperMoeLayer {
            layer: execution_layer,
            hyper_connection,
            attention,
            feed_forward: RoutedMoePayload {
                router: RouterWeights {
                    layer: execution_layer,
                    weight: linear("router.weight")?,
                    bias,
                    hash_table,
                    hash_rows,
                    hash_cols,
                },
                shared_expert: shared_ffn,
                router_policy: if !attachment && execution_layer < model.config.num_hash_layers {
                    ExpertRouterPolicy::sqrt_softplus_hash(
                        model.config.num_experts_per_tok,
                        model.config.route_scale,
                    )
                } else {
                    ExpertRouterPolicy::sqrt_softplus_score_topk(
                        model.config.num_experts_per_tok,
                        model.config.route_scale,
                    )
                },
            },
        })
    }

    #[cfg(feature = "cuda")]
    #[derive(Debug, Clone)]
    struct BoundCompressor {
        ape: BoundParameter,
        norm: BoundParameter,
        key_value: BoundParameter,
        gate: BoundParameter,
        ratio: usize,
        head_dim: usize,
        rotate: bool,
    }

    #[cfg(feature = "cuda")]
    fn materialize_compressed(
        materializer: &StateDictMaterializer,
        binding: &impl Fn(&str) -> Result<BoundParameter>,
        config: MlaConfig,
    ) -> Result<Option<MlaCompression>> {
        if config.compress_ratio == 0 {
            return Ok(None);
        }
        let compressor = materialize_compressor(
            materializer,
            &compressor_bindings(binding, "attention.compressor", config, false)?,
        )?;
        let indexer = if config.compress_ratio == 4 {
            let bindings =
                compressor_bindings(binding, "attention.indexer.compressor", config, true)?;
            Some(MlaIndexer {
                compressor: materialize_compressor(materializer, &bindings)?,
                wq_b: deepseek_linear(materializer, &binding("attention.indexer.query.weight")?)?,
                weights_proj: deepseek_linear(
                    materializer,
                    &binding("attention.indexer.weights.weight")?,
                )?,
            })
        } else {
            None
        };
        Ok(Some(MlaCompression {
            compressor,
            indexer,
        }))
    }

    #[cfg(feature = "cuda")]
    fn compressor_bindings(
        binding: &impl Fn(&str) -> Result<BoundParameter>,
        prefix: &str,
        config: MlaConfig,
        indexer: bool,
    ) -> Result<BoundCompressor> {
        let values = BoundCompressor {
            ape: binding(&format!("{prefix}.ape"))?,
            norm: binding(&format!("{prefix}.norm.weight"))?,
            key_value: binding(&format!("{prefix}.key_value.weight"))?,
            gate: binding(&format!("{prefix}.gate.weight"))?,
            ratio: config.compress_ratio,
            head_dim: if indexer {
                config.index_head_dim
            } else {
                config.head_dim
            },
            rotate: indexer,
        };
        same_residency([&values.ape, &values.norm, &values.key_value, &values.gate])?;
        Ok(values)
    }

    #[cfg(feature = "cuda")]
    fn materialize_compressor(
        materializer: &StateDictMaterializer,
        binding: &BoundCompressor,
    ) -> Result<MlaCompressor> {
        let ape = materializer.parameter(&binding.ape)?;
        let [ape_rows, ape_cols]: [usize; 2] = ape
            .binding()
            .weight()
            .logical_shape()
            .try_into()
            .map_err(|_| prepare_error("compressor APE is not a matrix"))?;
        Ok(MlaCompressor {
            compress_ratio: binding.ratio,
            head_dim: binding.head_dim,
            overlap: binding.ratio == 4,
            rotate_for_indexer: binding.rotate,
            ape: ape.values_f32()?,
            ape_rows,
            ape_cols,
            norm: materializer.parameter(&binding.norm)?.values_f32()?,
            wkv: deepseek_linear(materializer, &binding.key_value)?,
            wgate: deepseek_linear(materializer, &binding.gate)?,
        })
    }

    #[cfg(feature = "cuda")]
    fn materialize_proposal(
        model: &DeepSeekV4Checkpoint,
        materializer: &StateDictMaterializer,
    ) -> Result<Option<MtpAttachment<MlaHyperMoeLayer, Arc<ExpertSourceCatalog>>>> {
        let stage_count = model.resources().mtp_stage_count();
        if stage_count == 0 {
            return Ok(None);
        }
        let mut layers = Vec::with_capacity(stage_count);
        for stage in 0..stage_count {
            let prefix = format!("attachments.mtp.{stage}");
            let stage_path = path(&prefix)?;
            let bind = |suffix: &str| {
                model
                    .resources()
                    .decoder()
                    .require_attachment_path(&stage_path, &path(&format!("{prefix}.{suffix}"))?)
                    .cloned()
            };
            let execution_layer = model.config.num_layers + stage;
            layers.push(MtpStage {
                index: stage,
                execution_layer,
                backbone: materialize_layer(model, materializer, execution_layer, &prefix, true)?,
                main_projection: (stage == 0)
                    .then(|| deepseek_linear(materializer, &bind("main_projection.weight")?))
                    .transpose()?,
                main_norm: (stage == 0)
                    .then(|| {
                        norm_values(
                            materializer,
                            &bind("main_norm.weight")?,
                            model.config.norm_eps,
                        )
                    })
                    .transpose()?,
                experts: model
                    .resources()
                    .decoder()
                    .source_catalogs()
                    .experts_for_layer(execution_layer)?,
            });
        }
        let last = stage_count - 1;
        let stage_path = path(&format!("attachments.mtp.{last}"))?;
        let prefix = format!("{stage_path}.heads");
        let bind = |suffix: &str| {
            model
                .resources()
                .decoder()
                .require_attachment_path(&stage_path, &path(&format!("{prefix}.{suffix}"))?)
                .cloned()
        };
        let hc = [
            bind("hyper_connection.function")?,
            bind("hyper_connection.scale")?,
            bind("hyper_connection.base")?,
        ];
        let prediction_heads = MtpHeads {
            hyper_connection_head: HyperConnectionHead::materialize(
                materializer,
                [&hc[0], &hc[1], &hc[2]],
                model.config.hc_config(),
            )?,
            norm: norm_values(materializer, &bind("norm.weight")?, model.config.norm_eps)?,
            markov_embedding: deepseek_linear(materializer, &bind("markov_embedding.weight")?)?,
            markov_output: deepseek_linear(materializer, &bind("markov_output.weight")?)?,
            confidence: deepseek_linear(materializer, &bind("confidence.weight")?)?,
        };
        Ok(Some(MtpAttachment {
            stages: layers,
            heads: Some(prediction_heads),
            config: model.config.mtp_config(),
            hidden_size: model.config.hidden_size,
            norm_eps: model.config.norm_eps,
        }))
    }

    #[cfg(feature = "cuda")]
    fn norm_values(
        materializer: &StateDictMaterializer,
        binding: &BoundParameter,
        epsilon: f32,
    ) -> Result<Vec<f32>> {
        Ok(
            PreparedNorm::new(materializer.parameter(binding)?, epsilon)?
                .weight()
                .to_vec(),
        )
    }

    #[cfg(feature = "cuda")]
    fn build_kv_layout(
        model: &DeepSeekV4Checkpoint,
        max_layers: usize,
        proposal: Option<&MtpAttachment<MlaHyperMoeLayer, Arc<ExpertSourceCatalog>>>,
    ) -> Result<MlaKvLayout> {
        let target_configs = (0..max_layers)
            .map(|layer| model.config.attention_config_for_layer(layer))
            .collect::<Result<Vec<_>>>()?;
        let proposal_configs = proposal
            .into_iter()
            .flat_map(|attachment| attachment.stages.iter())
            .map(|stage| stage.backbone.attention.config());
        MlaKvLayout::new(target_configs.into_iter().chain(proposal_configs), 16)
    }

    #[cfg(feature = "cuda")]
    fn validate_options(
        model: &DeepSeekV4Checkpoint,
        options: DeepSeekV4PrepareOptions,
    ) -> Result<()> {
        if options.max_layers == 0 || options.max_layers > model.config.num_layers {
            return Err(prepare_error(format!(
                "max_layers must be in 1..={}, got {}",
                model.config.num_layers, options.max_layers
            )));
        }
        if options.output_head_chunk_rows == 0 || options.expert_reader_max_tensor_bytes == 0 {
            return Err(prepare_error("prepare capacities must be positive"));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn same_residency<const N: usize>(bindings: [&BoundParameter; N]) -> Result<()> {
        let first = bindings
            .first()
            .ok_or_else(|| prepare_error("empty typed binding group"))?;
        if bindings
            .iter()
            .skip(1)
            .any(|binding| binding.residency() != first.residency())
        {
            return Err(prepare_error("typed binding group mixes residencies"));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub(crate) use assembly::{DeepSeekV4ExecutionPolicy, DeepSeekV4Resources, prepare_with_policy};
#[cfg(feature = "cuda")]
pub use assembly::{DeepSeekV4ModelPlan, prepare};
