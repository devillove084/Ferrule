//! DeepSeek-V4 artifact model: HF weight loading and tensor binding.

use std::path::Path;

use crate::artifact::binding::{
    bind_attention_from_hf, bind_hyper_connection_from_hf, bind_hyper_connection_head_from_hf,
    bind_router_from_hf, bind_shared_swiglu_ffn_from_hf,
};
use crate::artifact::linear::ArtifactLinearPayload;
use crate::artifact::tensor::{ArtifactMatrixSlice, ArtifactTensorReader};
use crate::execution::ModelExecutionBackend;
use crate::hyper_connection::HyperConnectionHeadWeights;

use crate::moe::routing::ExpertRouterPolicy;
use crate::moe::streaming::{ExpertStreamingPlanner, ExpertStreamingPolicy};
use crate::runner::{ModelInfo, TokenLogit};
use crate::semantic::HyperConnectionStage;
use crate::tokenizer::TokenizerHandle;
use crate::{
    HfRoutedExpertTensorInfo, HfSafetensorsInventory, ModelDescriptor, ModelFamily, TensorRole,
    WeightSource,
};
use ferrule_common::{Error, Result};

use super::attention::{DeepSeekV4Attention, DeepSeekV4CompressedAttentionPayload};
use super::config::{
    DeepSeekV4Config, with_deepseek_v4_attention_execution_policies,
    with_deepseek_v4_swiglu_execution_policies,
};
use super::helpers::{
    decode_tensor_f32, decode_vector_f32, rank_logits_desc, read_named_vector_f32,
    unique_top_level_slice,
};
use super::layer::{DeepSeekV4Layer, DeepSeekV4LayerExpertRuntime, DeepSeekV4LayerState};
use super::operators::DeepSeekV4OperatorContext;

pub struct DeepSeekV4ArtifactModel {
    pub descriptor: ModelDescriptor,
    pub config: DeepSeekV4Config,
    pub tokenizer: TokenizerHandle,
    pub embedding: ArtifactMatrixSlice,
    pub output_norm: Vec<f32>,
    pub output_head: ArtifactMatrixSlice,
    pub hc_head: HyperConnectionHeadWeights,
    inventory: HfSafetensorsInventory,
    routed_expert_tensors_by_layer: Vec<Vec<HfRoutedExpertTensorInfo>>,
    max_tensor_bytes: u64,
}

impl DeepSeekV4ArtifactModel {
    pub fn load_hf_with_limit(model_dir: &Path, max_tensor_bytes: u64) -> Result<Self> {
        let descriptor = ModelDescriptor::load(model_dir)?;
        if descriptor.spec.family != ModelFamily::DeepSeekV4 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 artifact model expected DeepSeek-V4 descriptor, got {}",
                descriptor.spec.family
            )));
        }
        if descriptor.spec.weight_source != WeightSource::Safetensors {
            return Err(Error::Model(format!(
                "DeepSeek-V4 artifact model requires safetensors, got {}",
                descriptor.spec.weight_source
            )));
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
        let reader = ArtifactTensorReader::new(max_tensor_bytes);
        let tokenizer = TokenizerHandle::load(model_dir)?;
        let embedding = ArtifactMatrixSlice::from_slice(
            unique_top_level_slice(model_dir, &inventory, TensorRole::TokenEmbedding)?,
            "token embedding",
        )?;
        let output_norm = decode_vector_f32(&reader.read_slice(&unique_top_level_slice(
            model_dir,
            &inventory,
            TensorRole::OutputNorm,
        )?)?)?;
        let output_head = ArtifactMatrixSlice::from_slice(
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
            routed_expert_tensors_by_layer,
            max_tensor_bytes,
        })
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
            backend: "deepseek-v4-artifact",
        }
    }

    pub fn embedding_for_token(&self, token_id: u32) -> Result<Vec<f32>> {
        let token = token_id as usize;
        if token >= self.embedding.rows {
            return Err(Error::Model(format!(
                "DeepSeek-V4 token id {token_id} exceeds embedding vocab {}",
                self.embedding.rows
            )));
        }
        let reader = ArtifactTensorReader::new(self.max_tensor_bytes);
        let payload = reader.read_2d_rows(&self.embedding.slice, token, 1)?;
        let values = decode_tensor_f32(&payload)?;
        if values.len() != self.embedding.cols {
            return Err(Error::Model(format!(
                "DeepSeek-V4 embedding row length mismatch: expected {}, got {}",
                self.embedding.cols,
                values.len()
            )));
        }
        Ok(values)
    }

    pub fn initial_hc_state_for_token(&self, token_id: u32) -> Result<Vec<f32>> {
        let embedding = self.embedding_for_token(token_id)?;
        let mut state = Vec::with_capacity(self.config.hc_config().hc_hidden_size());
        for _ in 0..self.config.hc_mult {
            state.extend_from_slice(&embedding);
        }
        Ok(state)
    }

    pub fn initial_hc_state_for_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 HC state initialization requires at least one token".into(),
            ));
        }
        let hc_dim = self.config.hc_config().hc_hidden_size();
        let mut state = Vec::with_capacity(token_ids.len() * hc_dim);
        for &token_id in token_ids {
            let embedding = self.embedding_for_token(token_id)?;
            for _ in 0..self.config.hc_mult {
                state.extend_from_slice(&embedding);
            }
        }
        Ok(state)
    }

    pub fn normalized_hidden_from_hc_state(&self, hc_state: &[f32]) -> Result<Vec<f32>> {
        let mut operators = DeepSeekV4OperatorContext::new_cpu()?;
        self.normalized_hidden_from_hc_state_with_operators(hc_state, &mut operators)
    }

    pub(crate) fn normalized_hidden_from_hc_state_with_operators(
        &self,
        hc_state: &[f32],
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        self.normalized_last_hidden_from_hc_states_with_operators(hc_state, 1, operators)
    }

    pub(crate) fn normalized_last_hidden_from_hc_states_with_operators(
        &self,
        hc_state: &[f32],
        tokens: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        if tokens == 0 || hc_state.len() != tokens * self.config.hc_config().hc_hidden_size() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 HC head input mismatch: tokens={tokens} len={} expected {}",
                hc_state.len(),
                tokens * self.config.hc_config().hc_hidden_size()
            )));
        }
        let hidden = {
            #[cfg(feature = "cuda")]
            if operators.backend() == ModelExecutionBackend::Cuda {
                let state_buf = operators.cuda_mut()?.ops.upload_f32_buffer(hc_state)?;
                let hidden_buf = operators.cuda_mut()?.hc_head_from_device(
                    &state_buf,
                    tokens,
                    self.config.hc_config(),
                    &self.hc_head,
                )?;
                operators.cuda_mut()?.ops.download_f32_buffer(&hidden_buf)?
            } else {
                operators.hc_head(hc_state, tokens, self.config.hc_config(), &self.hc_head)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                operators.hc_head(hc_state, tokens, self.config.hc_config(), &self.hc_head)?
            }
        };
        let start = (tokens - 1) * self.config.hidden_size;
        operators.rms_norm(
            &hidden[start..start + self.config.hidden_size],
            &self.output_norm,
            self.config.norm_eps,
            "output_norm",
        )
    }

    pub fn logits_for_hidden_row_range(
        &self,
        hidden: &[f32],
        start_row: usize,
        row_count: usize,
    ) -> Result<Vec<f32>> {
        let mut operators = DeepSeekV4OperatorContext::new_cpu()?;
        self.logits_for_hidden_row_range_with_operators(
            hidden,
            start_row,
            row_count,
            &mut operators,
        )
    }

    pub(crate) fn logits_for_hidden_row_range_with_operators(
        &self,
        hidden: &[f32],
        start_row: usize,
        row_count: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        if hidden.len() != self.output_head.cols {
            return Err(Error::Model(format!(
                "DeepSeek-V4 output head input mismatch: expected {}, got {}",
                self.output_head.cols,
                hidden.len()
            )));
        }
        let reader = ArtifactTensorReader::new(self.max_tensor_bytes);
        let payload = reader.read_2d_rows(&self.output_head.slice, start_row, row_count)?;
        let linear =
            ArtifactLinearPayload::from_weight_and_scale(TensorRole::OutputHead, payload, None)?;
        operators.linear_matvec(&linear, hidden)
    }

    pub(crate) fn logits_for_hidden_chunked_with_operators(
        &self,
        hidden: &[f32],
        chunk_rows: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        if chunk_rows == 0 {
            return Err(Error::Model(
                "DeepSeek-V4 output head chunk_rows must be > 0".into(),
            ));
        }
        let mut logits = Vec::with_capacity(self.output_head.rows);
        let mut start = 0usize;
        while start < self.output_head.rows {
            let rows = chunk_rows.min(self.output_head.rows - start);
            logits.extend(
                self.logits_for_hidden_row_range_with_operators(hidden, start, rows, operators)?,
            );
            start += rows;
        }
        Ok(logits)
    }

    pub(crate) fn topk_logits_for_hidden_with_operators(
        &self,
        hidden: &[f32],
        top_k: usize,
        chunk_rows: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<TokenLogit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if chunk_rows == 0 {
            return Err(Error::Model(
                "DeepSeek-V4 output head chunk_rows must be > 0".into(),
            ));
        }
        let reader = ArtifactTensorReader::new(self.max_tensor_bytes);
        if operators.backend() == ModelExecutionBackend::Cuda {
            return operators.output_head_topk_chunks(
                &self.output_head.slice,
                hidden,
                top_k,
                chunk_rows,
                &reader,
            );
        }

        let mut top = Vec::<TokenLogit>::new();
        let mut start = 0usize;
        while start < self.output_head.rows {
            let rows = chunk_rows.min(self.output_head.rows - start);
            let payload = reader.read_2d_rows(&self.output_head.slice, start, rows)?;
            let linear = ArtifactLinearPayload::from_weight_and_scale(
                TensorRole::OutputHead,
                payload,
                None,
            )?;
            top.extend(
                operators
                    .linear_topk(&linear, hidden, top_k.min(rows))?
                    .into_iter()
                    .map(|mut item| {
                        item.token_id += start as u32;
                        item
                    }),
            );
            top.sort_by(rank_logits_desc);
            top.truncate(top_k);
            start += rows;
        }
        Ok(top)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn topk_logits_for_hidden_device_with_operators(
        &self,
        hidden: &ferrule_cuda::context::CudaF32Buffer,
        top_k: usize,
        chunk_rows: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<TokenLogit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if chunk_rows == 0 {
            return Err(Error::Model(
                "DeepSeek-V4 output head chunk_rows must be > 0".into(),
            ));
        }
        if hidden.len() != self.output_head.cols {
            return Err(Error::Model(format!(
                "DeepSeek-V4 output head device input mismatch: expected {}, got {}",
                self.output_head.cols,
                hidden.len()
            )));
        }
        let reader = ArtifactTensorReader::new(self.max_tensor_bytes);
        operators.cuda_mut()?.output_head_topk_chunks_with_device(
            &self.output_head.slice,
            hidden,
            top_k,
            chunk_rows,
            &reader,
        )
    }

    pub fn bind_layer(&self, layer: usize) -> Result<DeepSeekV4Layer> {
        let reader = ArtifactTensorReader::new(self.max_tensor_bytes);
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

    pub fn new_layer_sequence_state(&self, layer: usize) -> Result<DeepSeekV4LayerState> {
        let attention_config = self.config.attention_config_for_layer(layer)?;
        Ok(DeepSeekV4LayerState::new(attention_config))
    }

    pub fn new_layer_expert_runtime(
        &self,
        layer: usize,
        policy: ExpertStreamingPolicy,
    ) -> Result<DeepSeekV4LayerExpertRuntime> {
        let routed = self
            .routed_expert_tensors_by_layer
            .get(layer)
            .ok_or_else(|| Error::Model(format!("DeepSeek-V4 layer {layer} out of range")))?;
        let mut expert_planner = ExpertStreamingPlanner::new(policy);
        let registered = expert_planner
            .register_hf_routed_expert_tensor_sets(&self.descriptor.path, routed.iter().cloned())?;
        if registered != self.config.num_routed_experts {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} registered {registered} routed experts, expected {}",
                self.config.num_routed_experts
            )));
        }
        Ok(DeepSeekV4LayerExpertRuntime::new(expert_planner))
    }

    pub fn new_quality_first_layer_expert_runtime_with_residency(
        &self,
        layer: usize,
        moe_prefetch_experts: usize,
        moe_hotset_experts: usize,
    ) -> Result<DeepSeekV4LayerExpertRuntime> {
        let moe_hotset_experts = moe_hotset_experts.min(self.config.num_routed_experts);
        let moe_prefetch_experts = moe_prefetch_experts.min(self.config.num_routed_experts);
        let policy = if moe_hotset_experts > 0 {
            let gpu_slots_per_layer = moe_hotset_experts
                .max(self.config.num_experts_per_tok)
                .min(self.config.num_routed_experts);
            ExpertStreamingPolicy {
                gpu_slots_per_layer,
                prefetch_per_layer: moe_prefetch_experts
                    .min(gpu_slots_per_layer.saturating_sub(self.config.num_experts_per_tok)),
                preserve_artifact_quantization: true,
                allow_cpu_staging: true,
                allow_remote_sources: false,
            }
        } else if moe_prefetch_experts == 0 {
            ExpertStreamingPolicy {
                gpu_slots_per_layer: self.config.num_routed_experts,
                prefetch_per_layer: 0,
                preserve_artifact_quantization: true,
                allow_cpu_staging: true,
                allow_remote_sources: false,
            }
        } else {
            ExpertStreamingPolicy::quality_first_with_prefetch(
                self.config.num_experts_per_tok,
                moe_prefetch_experts,
            )
        };
        self.new_layer_expert_runtime(layer, policy)
    }
}
