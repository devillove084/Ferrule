//! DeepSeek-V4 recipe for the model-independent transformer graph.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::checkpoint::HfSafetensorsInventory;
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
