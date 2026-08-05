//! DeepSeek-V4 checkpoint: HF weight loading and tensor binding.

use std::path::Path;
use std::sync::Arc;

use super::checkpoint_binding::{
    bind_attention_from_hf, bind_hyper_connection_from_hf, bind_hyper_connection_head_from_hf,
    bind_router_from_hf, bind_shared_swiglu_ffn_from_hf,
};

use crate::checkpoint::tensor::{CheckpointMatrixSlice, CheckpointTensorReader};
use crate::hyper_connection::HyperConnectionHeadWeights;

use crate::moe::routing::ExpertRouterPolicy;
use crate::moe::streaming::{ExpertSourceCatalog, ExpertStreamingPolicy};
use crate::runner::ModelInfo;
use crate::semantic::HyperConnectionStage;
use crate::tokenizer::TokenizerHandle;
use crate::{HfSafetensorsInventory, ModelDescriptor, ModelFamily, TensorRole, WeightSource};
use ferrule_common::{Error, Result};

use super::attention::{DeepSeekV4Attention, DeepSeekV4CompressedAttentionPayload};
use super::config::{
    DeepSeekV4Config, with_deepseek_v4_attention_execution_policies,
    with_deepseek_v4_swiglu_execution_policies,
};
use super::helpers::{decode_vector_f32, read_named_vector_f32, unique_top_level_slice};
use super::layer::{DeepSeekV4Layer, DeepSeekV4LayerExpertRuntime, DeepSeekV4LayerState};
use super::proposal_attachment::{
    DeepSeekV4ProposalAttachment, load_proposal_heads, load_proposal_stage,
    parse_proposal_hyper_connection_tensor,
};

pub struct DeepSeekV4Checkpoint {
    pub descriptor: ModelDescriptor,
    pub config: DeepSeekV4Config,
    pub tokenizer: TokenizerHandle,
    pub embedding: CheckpointMatrixSlice,
    pub output_norm: Vec<f32>,
    pub output_head: CheckpointMatrixSlice,
    pub hc_head: HyperConnectionHeadWeights,
    inventory: HfSafetensorsInventory,
    routed_expert_catalogs_by_layer: Vec<Arc<ExpertSourceCatalog>>,
    max_tensor_bytes: u64,
}

impl DeepSeekV4Checkpoint {
    pub fn load_hf_with_limit(model_dir: &Path, max_tensor_bytes: u64) -> Result<Self> {
        let descriptor = ModelDescriptor::load(model_dir)?;
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
        let config = DeepSeekV4Config::from_hf_config(model_dir)?;
        let inventory = HfSafetensorsInventory::open(model_dir, ModelFamily::DeepSeekV4)?;
        let mut routed_expert_tensors_by_layer = vec![Vec::new(); config.num_layers];
        for tensor in inventory.routed_expert_tensors() {
            let layer = tensor.descriptor.layer;
            if layer < routed_expert_tensors_by_layer.len() {
                routed_expert_tensors_by_layer[layer].push(tensor);
            }
        }
        let routed_expert_catalogs_by_layer = routed_expert_tensors_by_layer
            .into_iter()
            .map(|tensors| {
                ExpertSourceCatalog::from_hf_routed_expert_tensor_sets(&descriptor.path, tensors)
                    .map(Arc::new)
            })
            .collect::<Result<Vec<_>>>()?;
        let reader = CheckpointTensorReader::new(max_tensor_bytes);
        let tokenizer = TokenizerHandle::load(model_dir)?;
        let embedding = CheckpointMatrixSlice::from_slice(
            unique_top_level_slice(model_dir, &inventory, TensorRole::TokenEmbedding)?,
            "token embedding",
        )?;
        let output_norm = decode_vector_f32(&reader.read_slice(&unique_top_level_slice(
            model_dir,
            &inventory,
            TensorRole::OutputNorm,
        )?)?)?;
        let output_head = CheckpointMatrixSlice::from_slice(
            unique_top_level_slice(model_dir, &inventory, TensorRole::OutputHead)?,
            "output head",
        )?;
        let hc_tensors = inventory.hyper_connection_tensors();
        let hc_head = bind_hyper_connection_head_from_hf(
            model_dir,
            &hc_tensors,
            &reader,
            config.hc_config(),
        )?;
        Ok(Self {
            descriptor,
            config,
            tokenizer,
            embedding,
            output_norm,
            output_head,
            hc_head,
            inventory,
            routed_expert_catalogs_by_layer,
            max_tensor_bytes,
        })
    }

    pub(crate) const fn inventory(&self) -> &HfSafetensorsInventory {
        &self.inventory
    }

    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            family: self.descriptor.spec.family.clone(),
            architecture: self.descriptor.spec.architecture.clone(),
            attention: self.descriptor.spec.attention.clone(),
            weight_source: self.descriptor.spec.weight_source,
            hidden_size: self.config.hidden_size,
            num_layers: self.config.num_layers,
            num_experts: self.config.num_routed_experts,
            num_experts_per_tok: self.config.num_experts_per_tok,
            vocab_size: self.config.vocab_size,
            backend: "deepseek-v4-checkpoint",
        }
    }

    pub fn bind_layer(&self, layer: usize) -> Result<DeepSeekV4Layer> {
        let reader = CheckpointTensorReader::new(self.max_tensor_bytes);
        let attention_tensors = self.inventory.attention_tensors();
        let hc_tensors = self.inventory.hyper_connection_tensors();
        let router_tensors = self.inventory.router_tensors();
        let shared_tensors = self.inventory.shared_expert_tensors();
        let attn_norm = read_named_vector_f32(
            &self.descriptor.path,
            &self.inventory,
            &reader,
            &format!("layers.{layer}.attn_norm.weight"),
            TensorRole::LayerNorm,
        )?;
        let ffn_norm = read_named_vector_f32(
            &self.descriptor.path,
            &self.inventory,
            &reader,
            &format!("layers.{layer}.ffn_norm.weight"),
            TensorRole::LayerNorm,
        )?;
        let attention_payload = with_deepseek_v4_attention_execution_policies(
            bind_attention_from_hf(&self.descriptor.path, layer, &attention_tensors, &reader)?,
        );
        let attention_config = self.config.attention_config_for_layer(layer)?;
        let compressed = DeepSeekV4CompressedAttentionPayload::bind_optional(
            layer,
            attention_config,
            &attention_payload.auxiliary,
            &reader,
        )?;
        let attention = DeepSeekV4Attention::new_with_compressed(
            layer,
            attention_config,
            attention_payload,
            compressed,
        )?;
        let hc_attention = bind_hyper_connection_from_hf(
            &self.descriptor.path,
            layer,
            HyperConnectionStage::Attention,
            &hc_tensors,
            &reader,
            self.config.hc_config(),
        )?;
        let hc_feed_forward = bind_hyper_connection_from_hf(
            &self.descriptor.path,
            layer,
            HyperConnectionStage::FeedForward,
            &hc_tensors,
            &reader,
            self.config.hc_config(),
        )?;
        let router = bind_router_from_hf(&self.descriptor.path, layer, &router_tensors, &reader)?;
        let shared_ffn =
            with_deepseek_v4_swiglu_execution_policies(bind_shared_swiglu_ffn_from_hf(
                &self.descriptor.path,
                layer,
                &shared_tensors,
                &reader,
                self.config.swiglu_limit,
            )?);
        let router_policy = if layer < self.config.num_hash_layers {
            ExpertRouterPolicy::sqrt_softplus_hash(
                self.config.num_experts_per_tok,
                self.config.route_scale,
            )
        } else {
            ExpertRouterPolicy::sqrt_softplus_score_topk(
                self.config.num_experts_per_tok,
                self.config.route_scale,
            )
        };
        Ok(DeepSeekV4Layer {
            layer,
            hc_config: self.config.hc_config(),
            attn_norm,
            ffn_norm,
            attention,
            hc_attention,
            hc_feed_forward,
            router,
            shared_ffn,
            router_policy,
        })
    }

    /// Loads the Proposal attachment stored under the checkpoint's `mtp.*` namespace.
    ///
    /// Returns `Ok(None)` when the checkpoint contains no Proposal attachment tensors.
    pub fn load_proposal_attachment(&self) -> Result<Option<DeepSeekV4ProposalAttachment>> {
        let proposal_tensors = self.inventory.mtp_layer_tensors();
        if proposal_tensors.is_empty() {
            return Ok(None);
        }
        let reader = CheckpointTensorReader::new(self.max_tensor_bytes);
        let hc_config = self.config.hc_config();
        let max_proposal_stage = *proposal_tensors
            .keys()
            .max()
            .expect("Proposal attachment tensors are non-empty");
        if proposal_tensors.len() != max_proposal_stage + 1
            || proposal_tensors.keys().copied().ne(0..=max_proposal_stage)
        {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 checkpoint Proposal stages are not contiguous from zero: {:?}",
                    proposal_tensors.keys().collect::<Vec<_>>()
                ),
            });
        }
        let mut layers = Vec::with_capacity(proposal_tensors.len());
        let mut prediction_heads = None;
        for (&proposal_stage, stage_tensors) in &proposal_tensors {
            let attention_config = self
                .config
                .attention_config_for_proposal_stage(proposal_stage)?;
            let stage = load_proposal_stage(
                &self.descriptor.path,
                proposal_stage,
                self.config.num_layers + proposal_stage,
                stage_tensors,
                &reader,
                attention_config,
                hc_config,
                self.config.swiglu_limit,
                self.config.num_experts_per_tok,
                self.config.route_scale,
            )?;
            if proposal_stage == max_proposal_stage {
                let hc_tensors: Vec<crate::checkpoint::inventory::HfHyperConnectionTensorInfo> =
                    stage_tensors
                        .iter()
                        .filter_map(parse_proposal_hyper_connection_tensor)
                        .collect();
                prediction_heads = Some(load_proposal_heads(
                    &self.descriptor.path,
                    proposal_stage,
                    stage_tensors,
                    &reader,
                    &hc_tensors,
                    hc_config,
                )?);
            }
            if stage.expert_source_catalog.count() != self.config.num_routed_experts {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 Proposal stage {proposal_stage} catalog has {} routed experts, expected {}",
                        stage.expert_source_catalog.count(),
                        self.config.num_routed_experts
                    ),
                });
            }
            layers.push(stage);
        }
        Ok(Some(DeepSeekV4ProposalAttachment {
            layers,
            prediction_heads,
            config: self.config.proposal_attachment_config(),
        }))
    }

    pub fn new_layer_sequence_state(&self, layer: usize) -> Result<DeepSeekV4LayerState> {
        let attention_config = self.config.attention_config_for_layer(layer)?;
        Ok(DeepSeekV4LayerState::new(attention_config))
    }

    pub fn expert_source_catalog(&self, layer: usize) -> Result<&Arc<ExpertSourceCatalog>> {
        self.routed_expert_catalogs_by_layer
            .get(layer)
            .ok_or_else(|| Error::Model {
                message: format!("DeepSeek-V4 layer {layer} out of range"),
            })
    }

    pub fn new_layer_expert_runtime(
        &self,
        layer: usize,
        policy: ExpertStreamingPolicy,
    ) -> Result<DeepSeekV4LayerExpertRuntime> {
        let catalog = self.expert_source_catalog(layer)?;
        if catalog.count() != self.config.num_routed_experts {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 layer {layer} catalog has {} routed experts, expected {}",
                    catalog.count(),
                    self.config.num_routed_experts
                ),
            });
        }
        Ok(DeepSeekV4LayerExpertRuntime::from_catalog(
            Arc::clone(catalog),
            policy,
        ))
    }

    pub fn resolved_expert_streaming_policy(
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

    pub fn new_layer_expert_runtime_with_residency(
        &self,
        layer: usize,
        moe_hotset_experts: usize,
    ) -> Result<DeepSeekV4LayerExpertRuntime> {
        self.new_layer_expert_runtime(
            layer,
            self.resolved_expert_streaming_policy(moe_hotset_experts),
        )
    }
}
