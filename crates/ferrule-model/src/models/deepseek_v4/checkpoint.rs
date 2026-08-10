//! Recipe-bound DeepSeek-V4 Hugging Face checkpoint.

use std::path::Path;

use ferrule_common::{Error, Result};

use crate::moe::streaming::ExpertStreamingPolicy;

use crate::tokenizer::TokenizerHandle;
use crate::transformer::{
    BoundDecoderResources, DecoderAttachmentSpec, DecoderRecipe, HFDecoderCheckpoint,
};
use crate::{
    AttentionKind, HfSafetensorsIndex, HfSafetensorsInventory, ModelDescriptor, ModelFamily,
    MoeSpec, QuantFormatCount, RouterKind, TransformerSemantics, TransformerSpec, WeightSource,
};

use super::config::DeepSeekV4Config;
use super::recipe::DeepSeekV4Recipe;

/// The sole DeepSeek view over a generic recipe-bound decoder state dict.
#[derive(Debug, Clone)]
pub struct BoundDeepSeekResources {
    decoder: BoundDecoderResources,
    mtp_stage_count: usize,
}

impl BoundDeepSeekResources {
    pub fn new(decoder: BoundDecoderResources) -> Result<Self> {
        let mtp_stage_count = decoder
            .spec()
            .attachments()
            .first()
            .map(|attachment| match attachment {
                DecoderAttachmentSpec::Proposal(proposal) => proposal.stages().len(),
            })
            .unwrap_or(0);
        Ok(Self {
            decoder,
            mtp_stage_count,
        })
    }

    pub const fn decoder(&self) -> &BoundDecoderResources {
        &self.decoder
    }

    pub const fn mtp_stage_count(&self) -> usize {
        self.mtp_stage_count
    }
}

/// Strict HF checkpoint whose only production binding is the DeepSeek-V4 recipe.
///
/// Opening the checkpoint validates the descriptor, builds the inventory-aware
/// recipe schema/name mapper, binds the complete state dict, and derives the
/// immutable source catalogs. Tensor payload bytes remain unread.
pub struct DeepSeekV4Checkpoint {
    pub(super) descriptor: ModelDescriptor,
    pub(super) config: DeepSeekV4Config,
    tokenizer: TokenizerHandle,
    resources: BoundDeepSeekResources,
    max_parameter_bytes: u64,
}

impl DeepSeekV4Checkpoint {
    pub fn load_hf_with_limit(model_dir: &Path, max_parameter_bytes: u64) -> Result<Self> {
        if max_parameter_bytes == 0 {
            return Err(Error::Model {
                message: "DeepSeek-V4 checkpoint parameter limit must be positive".into(),
            });
        }
        let config_json = read_config_json(model_dir)?;
        let config = DeepSeekV4Config::from_value(&config_json)?;
        let index_path = model_dir.join("model.safetensors.index.json");
        let index = HfSafetensorsIndex::open(&index_path)?;
        let missing_shards = index.missing_shards(model_dir);
        if !missing_shards.is_empty() {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 safetensors index references missing shards: {:?}",
                    missing_shards
                ),
            });
        }
        let inventory =
            HfSafetensorsInventory::from_index(model_dir, ModelFamily::DeepSeekV4, &index)?;
        let descriptor = descriptor_from_inventory(model_dir, &config, &inventory);
        validate_descriptor(&descriptor)?;

        // MTP stages are an artifact property, so the recipe and strict binding
        // reuse the one inventory discovered above.
        let output = DeepSeekV4Recipe::new()
            .build_with_inventory(&config_json, &inventory)
            .map_err(|source| Error::ModelSource {
                source: Box::new(source),
            })?;
        let (spec, schema, mapper) = output.into_parts();
        let checkpoint = HFDecoderCheckpoint::from_inventory(
            model_dir,
            index,
            inventory,
            spec,
            &schema,
            mapper.as_ref(),
        )?;
        let resources = BoundDeepSeekResources::new(checkpoint.into_resources())?;
        let tokenizer = TokenizerHandle::load(model_dir)?;

        Ok(Self {
            descriptor,
            config,
            tokenizer,
            resources,
            max_parameter_bytes,
        })
    }

    pub(super) const fn resources(&self) -> &BoundDeepSeekResources {
        &self.resources
    }

    pub(super) const fn max_parameter_bytes(&self) -> u64 {
        self.max_parameter_bytes
    }

    pub(super) fn into_tokenizer(self) -> TokenizerHandle {
        self.tokenizer
    }

    pub(super) fn resolved_expert_streaming_policy(
        &self,
        moe_hotset_experts: usize,
    ) -> ExpertStreamingPolicy {
        let moe_hotset_experts = moe_hotset_experts.min(self.config.num_routed_experts);
        let gpu_slots_per_layer = if moe_hotset_experts == 0 {
            self.config.num_routed_experts
        } else {
            moe_hotset_experts
                .max(self.config.num_experts_per_tok)
                .min(self.config.num_routed_experts)
        };
        ExpertStreamingPolicy {
            gpu_slots_per_layer,
            prefetch_per_layer: 0,
            preserve_source_encoding: true,
            allow_cpu_staging: true,
            allow_remote_sources: false,
        }
    }
}

fn descriptor_from_inventory(
    model_dir: &Path,
    config: &DeepSeekV4Config,
    inventory: &HfSafetensorsInventory,
) -> ModelDescriptor {
    ModelDescriptor {
        path: model_dir.to_path_buf(),
        spec: TransformerSpec {
            family: ModelFamily::DeepSeekV4,
            architecture: Some(config.architecture.clone()),
            weight_source: WeightSource::Safetensors,
            hidden_size: Some(config.hidden_size),
            num_layers: Some(config.num_layers),
            vocab_size: Some(config.vocab_size),
            num_heads: Some(config.num_heads),
            num_kv_heads: Some(1),
            head_dim: Some(config.head_dim),
            attention: AttentionKind::MultiLatentAttention,
            moe: MoeSpec {
                num_experts: Some(config.num_routed_experts),
                num_experts_per_tok: Some(config.num_experts_per_tok),
                has_shared_experts: true,
                router: if config.num_hash_layers == 0 {
                    RouterKind::DenseTopK
                } else {
                    RouterKind::HashAssistedTopK
                },
            },
            semantics: TransformerSemantics {
                norm_epsilon: Some(config.norm_eps),
                hyper_connection_epsilon: Some(config.hc_eps),
                hyper_connection_sinkhorn_iters: Some(config.hc_sinkhorn_iters),
                rope_theta: Some(config.rope_theta),
                rope_head_dim: Some(config.qk_rope_head_dim),
                rope_factor: Some(config.rope_factor),
                rope_original_max_position_embeddings: Some(config.original_seq_len),
                rope_beta_fast: Some(config.beta_fast),
                rope_beta_slow: Some(config.beta_slow),
                compress_rope_theta: Some(config.compress_rope_theta),
                attention_window_size: Some(config.window_size),
                attention_index_topk: Some(config.index_topk),
                attention_index_num_heads: Some(config.index_n_heads),
                attention_index_head_dim: Some(config.index_head_dim),
                attention_compress_ratios: config.compress_ratios.clone(),
                output_projection_groups: Some(config.o_groups),
                output_projection_rank: Some(config.o_lora_rank),
                swiglu_limit: Some(config.swiglu_limit),
                route_scale: Some(config.route_scale),
                num_hash_layers: Some(config.num_hash_layers),
            },
            tensor_count: Some(inventory.tensor_count),
            quantization: inventory
                .dtype_counts
                .iter()
                .map(|count| QuantFormatCount {
                    format: count.dtype.clone(),
                    tensors: count.tensors,
                })
                .collect(),
            notes: vec![format!(
                "HF safetensors header inventory: {} tensors, {} dtype classes, {} role classes",
                inventory.tensor_count,
                inventory.dtype_counts.len(),
                inventory.role_counts.len()
            )],
        },
        tensor_classes: inventory.class_counts.clone(),
    }
}

fn validate_descriptor(descriptor: &ModelDescriptor) -> Result<()> {
    if descriptor.spec.family != ModelFamily::DeepSeekV4 {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 checkpoint expected DeepSeek-V4 descriptor, got {}",
                descriptor.spec.family
            ),
        });
    }
    if descriptor.spec.weight_source != WeightSource::Safetensors {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 checkpoint requires safetensors, got {}",
                descriptor.spec.weight_source
            ),
        });
    }
    Ok(())
}

fn read_config_json(model_dir: &Path) -> Result<serde_json::Value> {
    let path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|error| {
        Error::context(
            format!("DeepSeek-V4 config '{}'", path.display()),
            error.into(),
        )
    })?;
    serde_json::from_str(&text).map_err(|source| Error::ModelSource {
        source: Box::new(source),
    })
}
