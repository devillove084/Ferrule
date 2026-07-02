//! DeepSeek-V4 model-specific runtime boundary.
//!
//! Keep DeepSeek-V4 forward semantics here. The surrounding runtime remains generic
//! over scheduling, sampling, source tensor IO, source linear formats, and CUDA
//! operator surfaces.

use std::cmp::Ordering;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::path::Path;

use ferrule_core::{Error, Result};
use ferrule_model::families::{deepseek_v4, HyperConnectionStage};
use ferrule_model::{
    HfSafetensorsInventory, HfSafetensorsTensorInfo, ModelDescriptor, ModelFamily, TensorRole,
    WeightSource,
};

use crate::attention_backend::{sparse_attention_reference, SparseAttentionSpec};
use crate::expert_executor::{CpuReferenceExpertExecutor, ExpertExecutor};
use crate::expert_handle::CpuExpertHandleStore;
#[cfg(feature = "cuda")]
use crate::expert_handle::{ExpertHandleStore, ExpertResidentFormat, ResidentExpertHandle};
use crate::expert_routing::ExpertRouterPolicy;
#[cfg(feature = "cuda")]
use crate::expert_streaming::{
    ExpertComputeBundle, ExpertId, ExpertLinearFormat, ExpertLinearPayload, ExpertStorageTier,
};
use crate::expert_streaming::{
    ExpertStreamingPlanner, ExpertStreamingPolicy, ExpertStreamingReader,
};
use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{
    hc_head_reference, hc_post_reference, hc_pre_reference, HyperConnectionConfig,
    HyperConnectionHeadWeights, HyperConnectionPreOutput, HyperConnectionSplit,
    HyperConnectionWeights,
};
use crate::routed_moe::{
    execute_routed_moe_with_source_router_reference_with_handles, RoutedMoeStepOutput,
};
use crate::runner::{ModelInfo, ModelRunner};
use crate::source_binding::{
    bind_attention_from_hf, bind_hyper_connection_from_hf, bind_hyper_connection_head_from_hf,
    bind_router_from_hf, bind_shared_swiglu_ffn_from_hf, AttentionSourcePayload,
    RouterSourcePayload,
};
#[cfg(feature = "cuda")]
use crate::source_linear::SourceLinearFormat;
use crate::source_linear::SourceLinearPayload;
use crate::source_tensor::{
    SourceDType, SourceTensorPayload, SourceTensorReader, SourceTensorSlice,
};
use crate::tokenizer::TokenizerHandle;

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4Config {
    pub hidden_size: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub norm_eps: f32,
    pub num_layers: usize,
    pub num_hash_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub vocab_size: usize,
    pub num_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub swiglu_limit: f32,
    pub route_scale: f32,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub original_seq_len: usize,
    pub rope_factor: f32,
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub compress_ratios: Vec<usize>,
}

impl DeepSeekV4Config {
    pub fn from_hf_config(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Model(format!("DeepSeek-V4 config '{}': {e}", path.display())))?;
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            Error::Model(format!("DeepSeek-V4 config json '{}': {e}", path.display()))
        })?;
        let rope_scaling = json.get("rope_scaling").unwrap_or(&serde_json::Value::Null);
        let compress_ratios = json
            .get("compress_ratios")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .map(|item| item.as_u64().map(|value| value as usize).unwrap_or(0))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![0; deepseek_v4::NUM_LAYERS]);

        Ok(Self {
            hidden_size: usize_key(&json, &["hidden_size"]).unwrap_or(deepseek_v4::HIDDEN_SIZE),
            hc_mult: usize_key(&json, &["hc_mult"]).unwrap_or(deepseek_v4::HC_MULT),
            hc_sinkhorn_iters: usize_key(&json, &["hc_sinkhorn_iters"])
                .unwrap_or(deepseek_v4::HC_SINKHORN_ITERS),
            hc_eps: f32_key(&json, &["hc_eps"]).unwrap_or(deepseek_v4::HC_EPS),
            norm_eps: f32_key(&json, &["rms_norm_eps", "norm_eps"])
                .unwrap_or(deepseek_v4::RMS_NORM_EPS),
            num_layers: usize_key(&json, &["num_hidden_layers", "n_layers"])
                .unwrap_or(deepseek_v4::NUM_LAYERS),
            num_hash_layers: usize_key(&json, &["num_hash_layers"])
                .unwrap_or(deepseek_v4::NUM_HASH_LAYERS),
            num_heads: usize_key(&json, &["num_attention_heads", "n_heads"])
                .unwrap_or(deepseek_v4::NUM_HEADS),
            head_dim: usize_key(&json, &["head_dim"]).unwrap_or(deepseek_v4::HEAD_DIM),
            q_lora_rank: usize_key(&json, &["q_lora_rank"]).unwrap_or(deepseek_v4::Q_LORA_RANK),
            qk_rope_head_dim: usize_key(&json, &["qk_rope_head_dim", "rope_head_dim"])
                .unwrap_or(deepseek_v4::QK_ROPE_HEAD_DIM),
            o_groups: usize_key(&json, &["o_groups"]).unwrap_or(deepseek_v4::O_GROUPS),
            o_lora_rank: usize_key(&json, &["o_lora_rank"]).unwrap_or(deepseek_v4::O_LORA_RANK),
            window_size: usize_key(&json, &["sliding_window", "window_size"])
                .unwrap_or(deepseek_v4::SLIDING_WINDOW),
            vocab_size: usize_key(&json, &["vocab_size"]).unwrap_or(deepseek_v4::VOCAB_SIZE),
            num_routed_experts: usize_key(&json, &["n_routed_experts", "num_experts"])
                .unwrap_or(deepseek_v4::N_ROUTED_EXPERTS),
            num_experts_per_tok: usize_key(&json, &["num_experts_per_tok"])
                .unwrap_or(deepseek_v4::NUM_EXPERTS_PER_TOK),
            moe_intermediate_size: usize_key(&json, &["moe_intermediate_size", "moe_inter_dim"])
                .unwrap_or(deepseek_v4::MOE_INTERMEDIATE_SIZE),
            swiglu_limit: f32_key(&json, &["swiglu_limit"]).unwrap_or(deepseek_v4::SWIGLU_LIMIT),
            route_scale: f32_key(&json, &["routed_scaling_factor", "route_scale"])
                .unwrap_or(deepseek_v4::ROUTED_SCALING_FACTOR),
            rope_theta: f32_key(&json, &["rope_theta"]).unwrap_or(deepseek_v4::ROPE_THETA),
            compress_rope_theta: f32_key(&json, &["compress_rope_theta"])
                .unwrap_or(deepseek_v4::COMPRESS_ROPE_THETA),
            original_seq_len: usize_key(rope_scaling, &["original_max_position_embeddings"])
                .or_else(|| usize_key(&json, &["original_seq_len"]))
                .unwrap_or(deepseek_v4::ORIGINAL_MAX_POSITION_EMBEDDINGS),
            rope_factor: f32_key(rope_scaling, &["factor"])
                .or_else(|| f32_key(&json, &["rope_factor"]))
                .unwrap_or(deepseek_v4::ROPE_FACTOR),
            beta_fast: usize_key(rope_scaling, &["beta_fast"])
                .or_else(|| usize_key(&json, &["beta_fast"]))
                .unwrap_or(deepseek_v4::ROPE_BETA_FAST),
            beta_slow: usize_key(rope_scaling, &["beta_slow"])
                .or_else(|| usize_key(&json, &["beta_slow"]))
                .unwrap_or(deepseek_v4::ROPE_BETA_SLOW),
            index_n_heads: usize_key(&json, &["index_n_heads"])
                .unwrap_or(deepseek_v4::INDEX_N_HEADS),
            index_head_dim: usize_key(&json, &["index_head_dim"])
                .unwrap_or(deepseek_v4::INDEX_HEAD_DIM),
            index_topk: usize_key(&json, &["index_topk"]).unwrap_or(deepseek_v4::INDEX_TOPK),
            compress_ratios,
        })
    }

    pub fn hc_config(&self) -> HyperConnectionConfig {
        HyperConnectionConfig {
            hc_mult: self.hc_mult,
            hidden_size: self.hidden_size,
            sinkhorn_iters: self.hc_sinkhorn_iters,
            eps: self.hc_eps,
            norm_eps: self.norm_eps,
        }
    }

    pub fn attention_config_for_layer(&self, layer: usize) -> Result<DeepSeekV4AttentionConfig> {
        if layer >= self.num_layers {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} exceeds layer count {}",
                self.num_layers
            )));
        }
        Ok(DeepSeekV4AttentionConfig {
            hidden_size: self.hidden_size,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            q_lora_rank: self.q_lora_rank,
            rope_head_dim: self.qk_rope_head_dim,
            o_groups: self.o_groups,
            o_lora_rank: self.o_lora_rank,
            window_size: self.window_size,
            compress_ratio: self.compress_ratios.get(layer).copied().unwrap_or(0),
            norm_eps: self.norm_eps,
            rope_theta: self.rope_theta,
            compress_rope_theta: self.compress_rope_theta,
            original_seq_len: self.original_seq_len,
            rope_factor: self.rope_factor,
            beta_fast: self.beta_fast,
            beta_slow: self.beta_slow,
            index_n_heads: self.index_n_heads,
            index_head_dim: self.index_head_dim,
            index_topk: self.index_topk,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceTensor2D {
    pub slice: SourceTensorSlice,
    pub rows: usize,
    pub cols: usize,
}

impl SourceTensor2D {
    pub fn from_slice(slice: SourceTensorSlice, label: &str) -> Result<Self> {
        let [rows, cols]: [usize; 2] =
            slice
                .shape
                .clone()
                .try_into()
                .map_err(|shape: Vec<usize>| {
                    Error::Model(format!(
                        "DeepSeek-V4 {label} '{}' expects 2D shape, got {:?}",
                        slice.name, shape
                    ))
                })?;
        Ok(Self { slice, rows, cols })
    }
}

pub struct DeepSeekV4SourceModel {
    pub descriptor: ModelDescriptor,
    pub config: DeepSeekV4Config,
    pub tokenizer: TokenizerHandle,
    pub embedding: SourceTensor2D,
    pub output_norm: Vec<f32>,
    pub output_head: SourceTensor2D,
    pub hc_head: HyperConnectionHeadWeights,
    inventory: HfSafetensorsInventory,
    max_tensor_bytes: u64,
}

impl DeepSeekV4SourceModel {
    pub fn load_hf(model_dir: &Path) -> Result<Self> {
        Self::load_hf_with_limit(model_dir, 128 * 1024 * 1024)
    }

    pub fn load_hf_with_limit(model_dir: &Path, max_tensor_bytes: u64) -> Result<Self> {
        let descriptor = ModelDescriptor::load(model_dir)?;
        if descriptor.spec.family != ModelFamily::DeepSeekV4 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 source model expected DeepSeek-V4 descriptor, got {}",
                descriptor.spec.family
            )));
        }
        if descriptor.spec.weight_source != WeightSource::Safetensors {
            return Err(Error::Model(format!(
                "DeepSeek-V4 source model requires safetensors, got {}",
                descriptor.spec.weight_source
            )));
        }
        let config = DeepSeekV4Config::from_hf_config(model_dir)?;
        let inventory = HfSafetensorsInventory::open(model_dir, ModelFamily::DeepSeekV4)?;
        let reader = SourceTensorReader::new(max_tensor_bytes);
        let tokenizer = TokenizerHandle::load(model_dir)?;
        let embedding = SourceTensor2D::from_slice(
            unique_top_level_slice(model_dir, &inventory, TensorRole::TokenEmbedding)?,
            "token embedding",
        )?;
        let output_norm = decode_vector_f32(&reader.read_slice(&unique_top_level_slice(
            model_dir,
            &inventory,
            TensorRole::OutputNorm,
        )?)?)?;
        let output_head = SourceTensor2D::from_slice(
            unique_top_level_slice(model_dir, &inventory, TensorRole::OutputHead)?,
            "output head",
        )?;
        let hc_tensors = inventory.hyper_connection_tensors(&ModelFamily::DeepSeekV4);
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
            backend: "deepseek-v4-source",
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
        let reader = SourceTensorReader::new(self.max_tensor_bytes);
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
        let mut operators = DeepSeekV4OperatorContext::new(DeepSeekV4OperatorBackend::Cpu)?;
        self.normalized_hidden_from_hc_state_with_operators(hc_state, &mut operators)
    }

    fn normalized_hidden_from_hc_state_with_operators(
        &self,
        hc_state: &[f32],
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        self.normalized_last_hidden_from_hc_states_with_operators(hc_state, 1, operators)
    }

    fn normalized_last_hidden_from_hc_states_with_operators(
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
        let hidden = operators.hc_head(hc_state, tokens, self.config.hc_config(), &self.hc_head)?;
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
        let mut operators = DeepSeekV4OperatorContext::new(DeepSeekV4OperatorBackend::Cpu)?;
        self.logits_for_hidden_row_range_with_operators(
            hidden,
            start_row,
            row_count,
            &mut operators,
        )
    }

    fn logits_for_hidden_row_range_with_operators(
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
        let reader = SourceTensorReader::new(self.max_tensor_bytes);
        let payload = reader.read_2d_rows(&self.output_head.slice, start_row, row_count)?;
        let linear =
            SourceLinearPayload::from_weight_and_scale(TensorRole::OutputHead, payload, None)?;
        operators.linear_matvec(&linear, hidden)
    }

    pub fn logits_for_hidden_chunked(&self, hidden: &[f32], chunk_rows: usize) -> Result<Vec<f32>> {
        let mut operators = DeepSeekV4OperatorContext::new(DeepSeekV4OperatorBackend::Cpu)?;
        self.logits_for_hidden_chunked_with_operators(hidden, chunk_rows, &mut operators)
    }

    fn logits_for_hidden_chunked_with_operators(
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

    pub fn topk_logits_for_hidden(
        &self,
        hidden: &[f32],
        top_k: usize,
        chunk_rows: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        let mut operators = DeepSeekV4OperatorContext::new(DeepSeekV4OperatorBackend::Cpu)?;
        self.topk_logits_for_hidden_with_operators(hidden, top_k, chunk_rows, &mut operators)
    }

    fn topk_logits_for_hidden_with_operators(
        &self,
        hidden: &[f32],
        top_k: usize,
        chunk_rows: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if chunk_rows == 0 {
            return Err(Error::Model(
                "DeepSeek-V4 output head chunk_rows must be > 0".into(),
            ));
        }
        let mut top = Vec::<DeepSeekV4Logit>::new();
        let reader = SourceTensorReader::new(self.max_tensor_bytes);
        let mut start = 0usize;
        while start < self.output_head.rows {
            let rows = chunk_rows.min(self.output_head.rows - start);
            let payload = reader.read_2d_rows(&self.output_head.slice, start, rows)?;
            let linear =
                SourceLinearPayload::from_weight_and_scale(TensorRole::OutputHead, payload, None)?;
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

    pub fn bind_layer(&self, layer: usize) -> Result<DeepSeekV4Layer> {
        let reader = SourceTensorReader::new(self.max_tensor_bytes);
        let attention_tensors = self.inventory.attention_tensors(&ModelFamily::DeepSeekV4);
        let hc_tensors = self
            .inventory
            .hyper_connection_tensors(&ModelFamily::DeepSeekV4);
        let router_tensors = self.inventory.router_tensors(&ModelFamily::DeepSeekV4);
        let shared_tensors = self
            .inventory
            .shared_expert_tensors(&ModelFamily::DeepSeekV4);
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
        let attention_payload =
            bind_attention_from_hf(&self.descriptor.path, layer, &attention_tensors, &reader)?;
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
        let shared_ffn = bind_shared_swiglu_ffn_from_hf(
            &self.descriptor.path,
            layer,
            &shared_tensors,
            &reader,
            self.config.swiglu_limit,
        )?;
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

    pub fn new_layer_state(
        &self,
        layer: usize,
        policy: ExpertStreamingPolicy,
    ) -> Result<DeepSeekV4LayerState> {
        let attention_config = self.config.attention_config_for_layer(layer)?;
        let routed = self
            .inventory
            .routed_expert_tensors(&ModelFamily::DeepSeekV4)
            .into_iter()
            .filter(|tensor| tensor.descriptor.layer == layer);
        let mut expert_planner = ExpertStreamingPlanner::new(policy);
        let registered =
            expert_planner.register_hf_routed_expert_tensor_sets(&self.descriptor.path, routed)?;
        if registered != self.config.num_routed_experts {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} registered {registered} routed experts, expected {}",
                self.config.num_routed_experts
            )));
        }
        Ok(DeepSeekV4LayerState {
            kv: DeepSeekV4AttentionCache::new(attention_config),
            expert_planner,
            expert_handles: CpuExpertHandleStore::new(),
        })
    }

    pub fn new_quality_first_layer_state(&self, layer: usize) -> Result<DeepSeekV4LayerState> {
        self.new_layer_state(
            layer,
            ExpertStreamingPolicy::quality_first(self.config.num_experts_per_tok),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepSeekV4Logit {
    pub token_id: u32,
    pub logit: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4OperatorBackend {
    /// CPU/reference execution for every operator. This is the correctness anchor.
    Cpu,
    /// CUDA execution for model operators. CPU remains only the explicit reference
    /// backend; unsupported CUDA formats fail instead of silently falling back.
    Cuda,
}

impl Default for DeepSeekV4OperatorBackend {
    fn default() -> Self {
        Self::Cpu
    }
}

impl DeepSeekV4OperatorBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu-reference",
            Self::Cuda => "cuda",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "cpu" | "cpu-reference" | "reference" => Ok(Self::Cpu),
            // Keep the old spelling as a temporary CLI compatibility alias; it now
            // means the same strict CUDA backend, not a CPU-fallback hybrid path.
            "cuda" | "cuda-hybrid" | "gpu" => Ok(Self::Cuda),
            other => Err(Error::Model(format!(
                "unknown DeepSeek-V4 operator backend '{other}' (expected cpu or cuda)"
            ))),
        }
    }
}

pub struct DeepSeekV4OperatorContext {
    backend: DeepSeekV4OperatorBackend,
    #[cfg(feature = "cuda")]
    cuda: Option<DeepSeekV4CudaOperatorCache>,
}

impl DeepSeekV4OperatorContext {
    pub fn new(backend: DeepSeekV4OperatorBackend) -> Result<Self> {
        Ok(Self {
            backend,
            #[cfg(feature = "cuda")]
            cuda: match backend {
                DeepSeekV4OperatorBackend::Cpu => None,
                DeepSeekV4OperatorBackend::Cuda => Some(DeepSeekV4CudaOperatorCache::new()?),
            },
        })
    }

    pub fn backend(&self) -> DeepSeekV4OperatorBackend {
        self.backend
    }

    fn linear_matvec(&mut self, linear: &SourceLinearPayload, input: &[f32]) -> Result<Vec<f32>> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => linear.reference_matvec(input),
            DeepSeekV4OperatorBackend::Cuda => self.cuda_matvec(linear, input),
        }
    }

    fn linear_topk(
        &mut self,
        linear: &SourceLinearPayload,
        input: &[f32],
        top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => {
                let logits = linear.reference_matvec(input)?;
                let mut top = logits
                    .into_iter()
                    .enumerate()
                    .map(|(token_id, logit)| DeepSeekV4Logit {
                        token_id: token_id as u32,
                        logit,
                    })
                    .collect::<Vec<_>>();
                top.sort_by(rank_logits_desc);
                top.truncate(top_k);
                Ok(top)
            }
            DeepSeekV4OperatorBackend::Cuda => self.cuda_linear_topk(linear, input, top_k),
        }
    }

    fn sparse_attention(
        &mut self,
        query: &[f32],
        values: &[f32],
        topk: &[isize],
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
        spec: SparseAttentionSpec,
    ) -> Result<Vec<f32>> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => {
                sparse_attention_reference(query, values, topk, Some(sink), tokens, kv_len, spec)
            }
            DeepSeekV4OperatorBackend::Cuda => {
                self.cuda_sparse_attention(query, values, topk, sink, tokens, kv_len, spec)
            }
        }
    }

    fn grouped_output_a(
        &mut self,
        output_a: &SourceLinearPayload,
        context: &[f32],
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<Vec<f32>> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => grouped_output_a(output_a, context, cfg, layer),
            DeepSeekV4OperatorBackend::Cuda => {
                self.cuda_grouped_output_a(output_a, context, cfg, layer)
            }
        }
    }

    fn rms_norm(
        &mut self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        label: &str,
    ) -> Result<Vec<f32>> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => rms_norm(input, weight, eps, label),
            DeepSeekV4OperatorBackend::Cuda => self.cuda_rms_norm(input, weight, eps),
        }
    }

    fn rms_norm_heads_in_place(
        &mut self,
        values: &mut [f32],
        heads: usize,
        head_dim: usize,
        eps: f32,
        layer: usize,
    ) -> Result<()> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => {
                rms_norm_heads_in_place(values, heads, head_dim, eps, layer)
            }
            DeepSeekV4OperatorBackend::Cuda => {
                let normalized = self.cuda_rms_norm_heads(values, heads, head_dim, eps)?;
                values.copy_from_slice(&normalized);
                Ok(())
            }
        }
    }

    fn hc_pre(
        &mut self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionWeights,
    ) -> Result<HyperConnectionPreOutput> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => hc_pre_reference(state, tokens, config, weights),
            DeepSeekV4OperatorBackend::Cuda => self.cuda_hc_pre(state, tokens, config, weights),
        }
    }

    fn hc_post(
        &mut self,
        hidden: &[f32],
        residual: &[f32],
        config: HyperConnectionConfig,
        split: &HyperConnectionSplit,
    ) -> Result<Vec<f32>> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => hc_post_reference(hidden, residual, config, split),
            DeepSeekV4OperatorBackend::Cuda => self.cuda_hc_post(hidden, residual, config, split),
        }
    }

    fn hc_head(
        &mut self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionHeadWeights,
    ) -> Result<Vec<f32>> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => hc_head_reference(state, tokens, config, weights),
            DeepSeekV4OperatorBackend::Cuda => self.cuda_hc_head(state, tokens, config, weights),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn routed_moe_step(
        &mut self,
        layer: usize,
        input: &[f32],
        token_id: u32,
        router: &RouterSourcePayload,
        predicted_experts: &[usize],
        router_policy: &ExpertRouterPolicy,
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
        expert_executor: &impl ExpertExecutor,
        shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<RoutedMoeStepOutput> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => {
                execute_routed_moe_with_source_router_reference_with_handles(
                    layer,
                    input,
                    token_id,
                    router,
                    predicted_experts,
                    router_policy,
                    planner,
                    reader,
                    handles,
                    expert_executor,
                    shared_expert,
                )
            }
            DeepSeekV4OperatorBackend::Cuda => self.cuda_routed_moe_step(
                layer,
                input,
                token_id,
                router,
                predicted_experts,
                router_policy,
                planner,
                reader,
                handles,
                shared_expert,
            ),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_mut(&mut self) -> Result<&mut DeepSeekV4CudaOperatorCache> {
        self.cuda.as_mut().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA operator cache is not initialized".into())
        })
    }

    #[cfg(feature = "cuda")]
    fn cuda_matvec(&mut self, linear: &SourceLinearPayload, input: &[f32]) -> Result<Vec<f32>> {
        self.cuda_mut()?.linear_matvec(linear, input)
    }

    #[cfg(feature = "cuda")]
    fn cuda_linear_topk(
        &mut self,
        linear: &SourceLinearPayload,
        input: &[f32],
        top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        self.cuda_mut()?.linear_topk(linear, input, top_k)
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_matvec(&mut self, _linear: &SourceLinearPayload, _input: &[f32]) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_sparse_attention(
        &mut self,
        query: &[f32],
        values: &[f32],
        topk: &[isize],
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
        spec: SparseAttentionSpec,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?
            .sparse_attention(query, values, topk, sink, tokens, kv_len, spec)
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_linear_topk(
        &mut self,
        _linear: &SourceLinearPayload,
        _input: &[f32],
        _top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_sparse_attention(
        &mut self,
        _query: &[f32],
        _values: &[f32],
        _topk: &[isize],
        _sink: &[f32],
        _tokens: usize,
        _kv_len: usize,
        _spec: SparseAttentionSpec,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_grouped_output_a(
        &mut self,
        output_a: &SourceLinearPayload,
        context: &[f32],
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?
            .grouped_output_a(output_a, context, cfg, layer)
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_grouped_output_a(
        &mut self,
        _output_a: &SourceLinearPayload,
        _context: &[f32],
        _cfg: DeepSeekV4AttentionConfig,
        _layer: usize,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_rms_norm(&mut self, input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
        self.cuda_mut()?.rms_norm(input, weight, eps)
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_rms_norm(&mut self, _input: &[f32], _weight: &[f32], _eps: f32) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_rms_norm_heads(
        &mut self,
        input: &[f32],
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?.rms_norm_heads(input, heads, head_dim, eps)
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_rms_norm_heads(
        &mut self,
        _input: &[f32],
        _heads: usize,
        _head_dim: usize,
        _eps: f32,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_hc_pre(
        &mut self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionWeights,
    ) -> Result<HyperConnectionPreOutput> {
        self.cuda_mut()?.hc_pre(state, tokens, config, weights)
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_hc_pre(
        &mut self,
        _state: &[f32],
        _tokens: usize,
        _config: HyperConnectionConfig,
        _weights: &HyperConnectionWeights,
    ) -> Result<HyperConnectionPreOutput> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_hc_post(
        &mut self,
        hidden: &[f32],
        residual: &[f32],
        config: HyperConnectionConfig,
        split: &HyperConnectionSplit,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?.hc_post(hidden, residual, config, split)
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_hc_post(
        &mut self,
        _hidden: &[f32],
        _residual: &[f32],
        _config: HyperConnectionConfig,
        _split: &HyperConnectionSplit,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_hc_head(
        &mut self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionHeadWeights,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?.hc_head(state, tokens, config, weights)
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_hc_head(
        &mut self,
        _state: &[f32],
        _tokens: usize,
        _config: HyperConnectionConfig,
        _weights: &HyperConnectionHeadWeights,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn cuda_routed_moe_step(
        &mut self,
        layer: usize,
        input: &[f32],
        token_id: u32,
        router: &RouterSourcePayload,
        predicted_experts: &[usize],
        router_policy: &ExpertRouterPolicy,
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
        shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<RoutedMoeStepOutput> {
        self.cuda_mut()?.routed_moe_step(
            layer,
            input,
            token_id,
            router,
            predicted_experts,
            router_policy,
            planner,
            reader,
            handles,
            shared_expert,
        )
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    fn cuda_routed_moe_step(
        &mut self,
        _layer: usize,
        _input: &[f32],
        _token_id: u32,
        _router: &RouterSourcePayload,
        _predicted_experts: &[usize],
        _router_policy: &ExpertRouterPolicy,
        _planner: &mut ExpertStreamingPlanner,
        _reader: &ExpertStreamingReader,
        _handles: &mut CpuExpertHandleStore,
        _shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<RoutedMoeStepOutput> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
struct DeepSeekV4CudaOperatorCache {
    ops: ferrule_cuda::context::CudaSourceOperatorContext,
    linears: HashMap<String, ferrule_cuda::context::CudaSourceLinearHandle>,
    experts: HashMap<ExpertId, CudaFp4ExpertHandles>,
}

#[cfg(feature = "cuda")]
struct CudaFp4ExpertHandles {
    gate: ferrule_cuda::context::CudaSourceLinearHandle,
    up: ferrule_cuda::context::CudaSourceLinearHandle,
    down: ferrule_cuda::context::CudaSourceLinearHandle,
    bytes: u64,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4CudaOperatorCache {
    fn new() -> Result<Self> {
        Ok(Self {
            ops: ferrule_cuda::context::CudaSourceOperatorContext::new()?,
            linears: HashMap::new(),
            experts: HashMap::new(),
        })
    }

    fn linear_matvec(&mut self, linear: &SourceLinearPayload, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != linear.format.in_features() {
            return Err(Error::Model(format!(
                "source linear {:?} input length mismatch: expected {}, got {}",
                linear.role,
                linear.format.in_features(),
                input.len()
            )));
        }
        let key = self.ensure_linear_uploaded(linear)?;
        let handle = self.linears.get(&key).expect("inserted above");
        self.ops.source_linear_matvec(handle, input)
    }

    fn linear_topk(
        &mut self,
        linear: &SourceLinearPayload,
        input: &[f32],
        top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        if input.len() != linear.format.in_features() {
            return Err(Error::Model(format!(
                "source linear {:?} top-k input length mismatch: expected {}, got {}",
                linear.role,
                linear.format.in_features(),
                input.len()
            )));
        }
        let key = self.ensure_linear_uploaded(linear)?;
        let handle = self.linears.get(&key).expect("inserted above");
        Ok(self
            .ops
            .source_linear_topk(handle, input, top_k)?
            .into_iter()
            .map(|(token_id, logit)| DeepSeekV4Logit { token_id, logit })
            .collect())
    }

    fn ensure_linear_uploaded(&mut self, linear: &SourceLinearPayload) -> Result<String> {
        let key = source_linear_cache_key(linear);
        if !self.linears.contains_key(&key) {
            let handle = self.upload_linear(linear)?;
            self.linears.insert(key.clone(), handle);
        }
        Ok(key)
    }

    fn upload_linear(
        &self,
        linear: &SourceLinearPayload,
    ) -> Result<ferrule_cuda::context::CudaSourceLinearHandle> {
        match linear.format {
            SourceLinearFormat::F32 {
                out_features,
                in_features,
            } => self
                .ops
                .upload_f32_linear(&linear.weight.bytes, out_features, in_features),
            SourceLinearFormat::Bf16 {
                out_features,
                in_features,
            } => self
                .ops
                .upload_bf16_linear(&linear.weight.bytes, out_features, in_features),
            SourceLinearFormat::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m,
                block_k,
            } => {
                let scale = linear.scale.as_ref().ok_or_else(|| {
                    Error::Model(format!(
                        "source linear {:?} CUDA FP8 weight is missing E8M0 scale tensor",
                        linear.role
                    ))
                })?;
                self.ops.upload_fp8_e4m3_e8m0_linear(
                    &linear.weight.bytes,
                    &scale.bytes,
                    out_features,
                    in_features,
                    block_m,
                    block_k,
                )
            }
            SourceLinearFormat::Fp4E2M1PackedWithE8M0Scale {
                out_features,
                in_features,
                block_size: 32,
            } => {
                let scale = linear.scale.as_ref().ok_or_else(|| {
                    Error::Model(format!(
                        "source linear {:?} CUDA FP4 weight is missing E8M0 scale tensor",
                        linear.role
                    ))
                })?;
                self.ops.upload_fp4_e2m1_e8m0_linear(
                    &linear.weight.bytes,
                    &scale.bytes,
                    out_features,
                    in_features,
                )
            }
            SourceLinearFormat::Fp4E2M1PackedWithE8M0Scale { block_size, .. } => {
                Err(Error::Model(format!(
                    "source linear {:?} CUDA FP4 block_size {block_size} is unsupported (expected 32)",
                    linear.role
                )))
            }
        }
    }

    fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
        self.ops.rms_norm(input, weight, eps)
    }

    fn rms_norm_heads(
        &self,
        input: &[f32],
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        self.ops.rms_norm_heads(input, heads, head_dim, eps)
    }

    fn swiglu_ffn(
        &mut self,
        ffn: &SwiGluFfnPayload,
        input: &[f32],
        output_scale: f32,
    ) -> Result<Vec<f32>> {
        let gate_key = self.ensure_linear_uploaded(&ffn.gate)?;
        let up_key = self.ensure_linear_uploaded(&ffn.up)?;
        let down_key = self.ensure_linear_uploaded(&ffn.down)?;
        let gate = self.linears.get(&gate_key).expect("inserted above");
        let up = self.linears.get(&up_key).expect("inserted above");
        let down = self.linears.get(&down_key).expect("inserted above");
        self.ops
            .source_swiglu_ffn_matvec(gate, up, down, input, output_scale, ffn.swiglu_limit)
    }

    fn hc_pre(
        &self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionWeights,
    ) -> Result<HyperConnectionPreOutput> {
        weights.validate(config)?;
        let (hidden, pre, post, comb) = self.ops.hc_pre_f32(
            state,
            &weights.function,
            &weights.scale,
            &weights.base,
            tokens,
            config.hc_mult,
            config.hidden_size,
            config.sinkhorn_iters,
            config.eps,
            config.norm_eps,
        )?;
        Ok(HyperConnectionPreOutput {
            hidden,
            split: HyperConnectionSplit {
                tokens,
                hc_mult: config.hc_mult,
                pre,
                post,
                comb,
            },
        })
    }

    fn hc_post(
        &self,
        hidden: &[f32],
        residual: &[f32],
        config: HyperConnectionConfig,
        split: &HyperConnectionSplit,
    ) -> Result<Vec<f32>> {
        self.ops.hc_post_f32(
            hidden,
            residual,
            &split.post,
            &split.comb,
            split.tokens,
            config.hc_mult,
            config.hidden_size,
        )
    }

    fn hc_head(
        &self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionHeadWeights,
    ) -> Result<Vec<f32>> {
        weights.validate(config)?;
        self.ops.hc_head_f32(
            state,
            &weights.function,
            &weights.scale,
            &weights.base,
            tokens,
            config.hc_mult,
            config.hidden_size,
            config.eps,
            config.norm_eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn routed_moe_step(
        &mut self,
        layer: usize,
        input: &[f32],
        token_id: u32,
        router: &RouterSourcePayload,
        predicted_experts: &[usize],
        router_policy: &ExpertRouterPolicy,
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
        shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<RoutedMoeStepOutput> {
        let logits = self.linear_matvec(&router.weight, input)?;
        let hash_experts = router.hash_experts_for_token(token_id)?;
        let routes =
            router_policy.route(&logits, router.bias.as_deref(), hash_experts.as_deref())?;
        let selected = routes.iter().map(|route| route.expert).collect::<Vec<_>>();
        let streaming = planner.plan_layer_step(layer, &selected, predicted_experts)?;

        handles.apply_evictions(&streaming.evictions);
        for eviction in &streaming.evictions {
            self.experts.remove(&eviction.expert);
        }

        for load in &streaming.loads {
            let payload = reader.read_source(load.expert, &load.source)?;
            let bundle = ExpertComputeBundle::from_source_payload(payload)?;
            let expert = self.upload_expert_bundle(&bundle)?;
            let bytes = expert.bytes;
            self.experts.insert(load.expert, expert);
            handles.insert_resident_handle(ResidentExpertHandle::new(
                load.expert,
                ExpertStorageTier::Gpu,
                ExpertResidentFormat::Opaque("cuda-fp4-source".into()),
                bytes,
            ))?;
        }

        let mut routed_output = None::<Vec<f32>>;
        for route in &routes {
            let expert_id = ExpertId::new(layer, route.expert);
            let expert = self.experts.get(&expert_id).ok_or_else(|| {
                Error::Model(format!(
                    "CUDA expert handle missing for layer {} expert {}",
                    expert_id.layer, expert_id.expert
                ))
            })?;
            let expert_out = self.ops.source_fp4_swiglu_ffn_matvec(
                &expert.gate,
                &expert.up,
                &expert.down,
                input,
                route.weight,
                shared_expert.map(|ffn| ffn.swiglu_limit).unwrap_or(0.0),
            )?;
            accumulate_output(&mut routed_output, expert_out)?;
        }
        let routed_output = routed_output.unwrap_or_else(|| vec![0.0; input.len()]);
        let shared_output = shared_expert
            .map(|shared| self.swiglu_ffn(shared, input, 1.0))
            .transpose()?;
        let mut output = routed_output.clone();
        if let Some(shared) = &shared_output {
            if shared.len() != output.len() {
                return Err(Error::Model(format!(
                    "shared expert output length mismatch: routed={}, shared={}",
                    output.len(),
                    shared.len()
                )));
            }
            for (dst, value) in output.iter_mut().zip(shared.iter()) {
                *dst += value;
            }
        }
        planner.commit_step(&streaming)?;

        Ok(RoutedMoeStepOutput {
            routes,
            streaming,
            routed_output,
            shared_output,
            output,
        })
    }

    fn upload_expert_bundle(&self, bundle: &ExpertComputeBundle) -> Result<CudaFp4ExpertHandles> {
        Ok(CudaFp4ExpertHandles {
            gate: self.upload_expert_linear(&bundle.gate)?,
            up: self.upload_expert_linear(&bundle.up)?,
            down: self.upload_expert_linear(&bundle.down)?,
            bytes: bundle.total_bytes(),
        })
    }

    fn upload_expert_linear(
        &self,
        linear: &ExpertLinearPayload,
    ) -> Result<ferrule_cuda::context::CudaSourceLinearHandle> {
        let ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features,
            in_features,
            block_size: 32,
        } = linear.format
        else {
            return Err(Error::Model(format!(
                "CUDA routed expert {:?} requires source FP4 block_size=32, got {:?}",
                linear.matrix, linear.format
            )));
        };
        let scale = linear.scale.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "CUDA routed expert {:?} is missing E8M0 scale payload",
                linear.matrix
            ))
        })?;
        self.ops.upload_fp4_e2m1_e8m0_linear(
            &linear.weight.bytes,
            &scale.bytes,
            out_features,
            in_features,
        )
    }

    fn sparse_attention(
        &mut self,
        query: &[f32],
        values: &[f32],
        topk: &[isize],
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
        spec: SparseAttentionSpec,
    ) -> Result<Vec<f32>> {
        let topk_i32 = topk
            .iter()
            .map(|&idx| {
                i32::try_from(idx).map_err(|_| {
                    Error::Model(format!("sparse top-k index {idx} exceeds i32 CUDA ABI"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let shape = ferrule_cuda::transformer::sparse_attention::CudaSparseAttentionShape {
            batch_size: 1,
            tokens_per_batch: tokens,
            kv_len,
            heads: spec.heads,
            head_dim: spec.head_dim,
            topk: spec.topk,
            softmax_scale: spec.softmax_scale,
        };
        self.ops
            .sparse_attention_sink_f32(query, values, &topk_i32, sink, shape)
    }

    fn grouped_output_a(
        &mut self,
        output_a: &SourceLinearPayload,
        context: &[f32],
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<Vec<f32>> {
        let SourceLinearFormat::Fp8E4M3WithE8M0Scale {
            out_features,
            in_features,
            block_m,
            block_k,
        } = output_a.format
        else {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA grouped wo_a requires FP8 source format, got {:?}",
                output_a.format
            )));
        };
        if out_features != cfg.output_latent_dim() || in_features != cfg.output_group_input_dim() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA grouped wo_a shape mismatch: got [{out_features}, {in_features}], expected [{}, {}]",
                cfg.output_latent_dim(),
                cfg.output_group_input_dim()
            )));
        }
        if cfg.o_lora_rank % block_m != 0 || cfg.output_group_input_dim() % block_k != 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA grouped wo_a block shape unsupported: o_rank={} group_in={} block_m={block_m} block_k={block_k}",
                cfg.o_lora_rank,
                cfg.output_group_input_dim()
            )));
        }
        if context.len() != cfg.q_full_dim() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} context length mismatch: expected {}, got {}",
                cfg.q_full_dim(),
                context.len()
            )));
        }
        let scale = output_a.scale.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} grouped wo_a FP8 weight missing scale tensor"
            ))
        })?;
        let group_in = cfg.output_group_input_dim();
        let scale_cols = group_in.div_ceil(block_k);
        let scale_rows_per_group = cfg.o_lora_rank.div_ceil(block_m);
        let mut out = vec![0.0f32; cfg.output_latent_dim()];
        for group in 0..cfg.o_groups {
            let row_start = group * cfg.o_lora_rank;
            let weight_start = row_start * group_in;
            let weight_end = weight_start + cfg.o_lora_rank * group_in;
            let scale_start = (row_start / block_m) * scale_cols;
            let scale_end = scale_start + scale_rows_per_group * scale_cols;
            let key = format!("{}::grouped_wo_a::{group}", output_a.weight.slice.name);
            if !self.linears.contains_key(&key) {
                let handle = self.ops.upload_fp8_e4m3_e8m0_linear(
                    &output_a.weight.bytes[weight_start..weight_end],
                    &scale.bytes[scale_start..scale_end],
                    cfg.o_lora_rank,
                    group_in,
                    block_m,
                    block_k,
                )?;
                self.linears.insert(key.clone(), handle);
            }
            let handle = self.linears.get(&key).expect("inserted above");
            let context_start = group * group_in;
            let group_out = self
                .ops
                .source_linear_matvec(handle, &context[context_start..context_start + group_in])?;
            out[row_start..row_start + cfg.o_lora_rank].copy_from_slice(&group_out);
        }
        Ok(out)
    }
}

#[cfg(feature = "cuda")]
fn source_linear_cache_key(linear: &SourceLinearPayload) -> String {
    let scale_name = linear
        .scale
        .as_ref()
        .map(|scale| scale.slice.name.as_str())
        .unwrap_or("<none>");
    format!(
        "{}::{scale_name}::{:?}",
        linear.weight.slice.name, linear.format
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4ReferenceOptions {
    pub max_layers: usize,
    pub output_head_chunk_rows: usize,
    pub expert_reader_max_tensor_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4LayerRuntimeStats {
    pub layer: usize,
    pub window_kv_len: usize,
    pub compressed_kv_len: usize,
    pub indexer_compressed_kv_len: usize,
    pub resident_experts: usize,
    pub resident_expert_bytes: u64,
}

impl Default for DeepSeekV4ReferenceOptions {
    fn default() -> Self {
        Self {
            max_layers: deepseek_v4::NUM_LAYERS,
            output_head_chunk_rows: 1024,
            expert_reader_max_tensor_bytes: 64 * 1024 * 1024,
        }
    }
}

pub struct DeepSeekV4ReferenceRunner {
    pub model: DeepSeekV4SourceModel,
    pub options: DeepSeekV4ReferenceOptions,
    operators: DeepSeekV4OperatorContext,
    layers: Vec<Option<DeepSeekV4Layer>>,
    states: Vec<Option<DeepSeekV4LayerState>>,
    position: usize,
    expert_reader: ExpertStreamingReader,
    expert_executor: CpuReferenceExpertExecutor,
}

impl DeepSeekV4ReferenceRunner {
    pub fn new(model: DeepSeekV4SourceModel, options: DeepSeekV4ReferenceOptions) -> Result<Self> {
        Self::new_with_operator_backend(model, options, DeepSeekV4OperatorBackend::Cpu)
    }

    pub fn new_with_operator_backend(
        model: DeepSeekV4SourceModel,
        options: DeepSeekV4ReferenceOptions,
        operator_backend: DeepSeekV4OperatorBackend,
    ) -> Result<Self> {
        if options.max_layers > model.config.num_layers {
            return Err(Error::Model(format!(
                "DeepSeek-V4 reference runner max_layers {} exceeds model layers {}",
                options.max_layers, model.config.num_layers
            )));
        }
        if options.output_head_chunk_rows == 0 {
            return Err(Error::Model(
                "DeepSeek-V4 reference runner output_head_chunk_rows must be > 0".into(),
            ));
        }
        let mut layers = Vec::new();
        layers.resize_with(options.max_layers, || None);
        let mut states = Vec::new();
        states.resize_with(options.max_layers, || None);
        let swiglu_limit = model.config.swiglu_limit;
        Ok(Self {
            model,
            options,
            operators: DeepSeekV4OperatorContext::new(operator_backend)?,
            layers,
            states,
            position: 0,
            expert_reader: ExpertStreamingReader::new(options.expert_reader_max_tensor_bytes),
            expert_executor: CpuReferenceExpertExecutor::new(swiglu_limit),
        })
    }

    pub fn load_hf_with_options(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: DeepSeekV4ReferenceOptions,
    ) -> Result<Self> {
        Self::new(
            DeepSeekV4SourceModel::load_hf_with_limit(model_dir, max_tensor_bytes)?,
            options,
        )
    }

    pub fn load_hf_with_options_and_backend(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: DeepSeekV4ReferenceOptions,
        operator_backend: DeepSeekV4OperatorBackend,
    ) -> Result<Self> {
        Self::new_with_operator_backend(
            DeepSeekV4SourceModel::load_hf_with_limit(model_dir, max_tensor_bytes)?,
            options,
            operator_backend,
        )
    }

    pub fn operator_backend(&self) -> DeepSeekV4OperatorBackend {
        self.operators.backend()
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn reset(&mut self) {
        for state in &mut self.states {
            *state = None;
        }
        self.position = 0;
    }

    pub fn bound_layer_count(&self) -> usize {
        self.layers.iter().filter(|layer| layer.is_some()).count()
    }

    pub fn layer_runtime_stats(&self) -> Vec<DeepSeekV4LayerRuntimeStats> {
        let mut stats = Vec::new();
        for layer_idx in 0..self.options.max_layers {
            let (Some(layer), Some(state)) = (&self.layers[layer_idx], &self.states[layer_idx])
            else {
                continue;
            };
            let index_head_dim = layer.attention.config.index_head_dim;
            stats.push(DeepSeekV4LayerRuntimeStats {
                layer: layer_idx,
                window_kv_len: state.kv.len(),
                compressed_kv_len: state.kv.compressed_len(),
                indexer_compressed_kv_len: state.kv.indexer_compressed_len(index_head_dim),
                resident_experts: state.expert_handles.len(),
                resident_expert_bytes: state.expert_handles.total_bytes(),
            });
        }
        stats
    }

    pub fn decode_token_hidden(&mut self, token_id: u32) -> Result<Vec<f32>> {
        let mut hc_state = self.model.initial_hc_state_for_token(token_id)?;
        for layer_idx in 0..self.options.max_layers {
            if self.layers[layer_idx].is_none() {
                self.layers[layer_idx] = Some(self.model.bind_layer(layer_idx)?);
            }
            if self.states[layer_idx].is_none() {
                self.states[layer_idx] = Some(self.model.new_quality_first_layer_state(layer_idx)?);
            }
            let layer = self.layers[layer_idx].as_ref().expect("initialized above");
            let state = self.states[layer_idx].as_mut().expect("initialized above");
            let step = layer.decode_step_with_operators(
                state,
                &hc_state,
                token_id,
                self.position,
                &[],
                &self.expert_reader,
                &self.expert_executor,
                &mut self.operators,
            )?;
            hc_state = step.hc_state;
        }
        self.position += 1;
        self.model
            .normalized_hidden_from_hc_state_with_operators(&hc_state, &mut self.operators)
    }

    pub fn decode_token_logits_row_range(
        &mut self,
        token_id: u32,
        start_row: usize,
        row_count: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.decode_token_hidden(token_id)?;
        self.model.logits_for_hidden_row_range_with_operators(
            &hidden,
            start_row,
            row_count,
            &mut self.operators,
        )
    }

    pub fn decode_token_logits(&mut self, token_id: u32) -> Result<Vec<f32>> {
        let hidden = self.decode_token_hidden(token_id)?;
        self.model.logits_for_hidden_chunked_with_operators(
            &hidden,
            self.options.output_head_chunk_rows,
            &mut self.operators,
        )
    }

    pub fn decode_token_topk(
        &mut self,
        token_id: u32,
        top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        let hidden = self.decode_token_hidden(token_id)?;
        self.model.topk_logits_for_hidden_with_operators(
            &hidden,
            top_k,
            self.options.output_head_chunk_rows,
            &mut self.operators,
        )
    }

    /// Advance the model session with a token without materializing lm_head logits.
    pub fn feed_token(&mut self, token_id: u32) -> Result<()> {
        self.decode_token_hidden(token_id).map(|_| ())
    }

    /// Correctness-first prefill fallback: execute prompt tokens one-by-one through
    /// the decode path while preserving KV/compressor state. This is intentionally
    /// not the production batched prefill kernel, but it exercises real DSV4 weights
    /// and keeps semantics close to the official causal path.
    pub fn prefill_tokens_hidden(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 prefill requires at least one token".into(),
            ));
        }
        let mut hidden = Vec::new();
        for &token_id in token_ids {
            hidden = self.decode_token_hidden(token_id)?;
        }
        Ok(hidden)
    }

    pub fn prefill_tokens_logits_row_range(
        &mut self,
        token_ids: &[u32],
        start_row: usize,
        row_count: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.prefill_tokens_hidden(token_ids)?;
        self.model.logits_for_hidden_row_range_with_operators(
            &hidden,
            start_row,
            row_count,
            &mut self.operators,
        )
    }

    pub fn prefill_tokens_logits(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.prefill_tokens_hidden(token_ids)?;
        self.model.logits_for_hidden_chunked_with_operators(
            &hidden,
            self.options.output_head_chunk_rows,
            &mut self.operators,
        )
    }

    pub fn prefill_tokens_topk(
        &mut self,
        token_ids: &[u32],
        top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        let hidden = self.prefill_tokens_hidden(token_ids)?;
        self.model.topk_logits_for_hidden_with_operators(
            &hidden,
            top_k,
            self.options.output_head_chunk_rows,
            &mut self.operators,
        )
    }

    /// Public prefill entry point for chat/run integration.
    ///
    /// At session start this follows the official DSV4 prefill shape: each layer sees
    /// the full prompt segment with `start_pos == 0`, builds window/compressed/indexer
    /// KV in bulk, and only then advances to the next layer. For non-zero session
    /// positions we keep the verified sequential append path until a paged/persistent
    /// multi-turn prefill cache is added.
    pub fn prefill_tokens_hidden_batched(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 prefill requires at least one token".into(),
            ));
        }
        if self.position != 0 || token_ids.len() == 1 {
            return self.prefill_tokens_hidden(token_ids);
        }

        let tokens = token_ids.len();
        let mut hc_state = self.model.initial_hc_state_for_tokens(token_ids)?;
        for layer_idx in 0..self.options.max_layers {
            if self.layers[layer_idx].is_none() {
                self.layers[layer_idx] = Some(self.model.bind_layer(layer_idx)?);
            }
            if self.states[layer_idx].is_none() {
                self.states[layer_idx] = Some(self.model.new_quality_first_layer_state(layer_idx)?);
            }
            let layer = self.layers[layer_idx].as_ref().expect("initialized above");
            let state = self.states[layer_idx].as_mut().expect("initialized above");
            hc_state = layer.prefill_start_with_operators(
                state,
                &hc_state,
                token_ids,
                self.position,
                &[],
                &self.expert_reader,
                &self.expert_executor,
                &mut self.operators,
            )?;
        }
        self.position += tokens;
        self.model
            .normalized_last_hidden_from_hc_states_with_operators(
                &hc_state,
                tokens,
                &mut self.operators,
            )
    }

    pub fn prefill_tokens_logits_batched(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.prefill_tokens_hidden_batched(token_ids)?;
        self.model.logits_for_hidden_chunked_with_operators(
            &hidden,
            self.options.output_head_chunk_rows,
            &mut self.operators,
        )
    }

    pub fn prefill_tokens_logits_row_range_batched(
        &mut self,
        token_ids: &[u32],
        start_row: usize,
        row_count: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.prefill_tokens_hidden_batched(token_ids)?;
        self.model.logits_for_hidden_row_range_with_operators(
            &hidden,
            start_row,
            row_count,
            &mut self.operators,
        )
    }

    pub fn prefill_tokens_topk_batched(
        &mut self,
        token_ids: &[u32],
        top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        let hidden = self.prefill_tokens_hidden_batched(token_ids)?;
        self.model.topk_logits_for_hidden_with_operators(
            &hidden,
            top_k,
            self.options.output_head_chunk_rows,
            &mut self.operators,
        )
    }
}

impl ModelRunner for DeepSeekV4ReferenceRunner {
    fn model_info(&self) -> ModelInfo {
        let mut info = self.model.model_info();
        info.backend = self.operator_backend().as_str();
        info
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.model.tokenizer.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.model.tokenizer.decode(tokens)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.prefill_tokens_logits_batched(tokens)
    }

    fn decode_token(&mut self, token: u32) -> Result<Vec<f32>> {
        self.decode_token_logits(token)
    }

    fn reset_session(&mut self) -> Result<()> {
        self.reset();
        Ok(())
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.model.tokenizer.eos_token_id()
    }

    fn expert_report(&self) -> Option<String> {
        let stats = self.layer_runtime_stats();
        if stats.is_empty() {
            return Some("DeepSeek-V4 layers are not bound yet.\n".into());
        }
        let mut report = String::new();
        for stat in stats {
            report.push_str(&format!(
                "L{:>2}: window_kv={} compressed_kv={} indexer_kv={} resident_experts={} resident_bytes={}\n",
                stat.layer,
                stat.window_kv_len,
                stat.compressed_kv_len,
                stat.indexer_compressed_kv_len,
                stat.resident_experts,
                stat.resident_expert_bytes
            ));
        }
        Some(report)
    }
}

pub struct DeepSeekV4Layer {
    pub layer: usize,
    pub hc_config: HyperConnectionConfig,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub attention: DeepSeekV4Attention,
    pub hc_attention: HyperConnectionWeights,
    pub hc_feed_forward: HyperConnectionWeights,
    pub router: RouterSourcePayload,
    pub shared_ffn: SwiGluFfnPayload,
    pub router_policy: ExpertRouterPolicy,
}

impl DeepSeekV4Layer {
    pub fn decode_step_reference(
        &self,
        state: &mut DeepSeekV4LayerState,
        hc_state: &[f32],
        token_id: u32,
        position: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        expert_executor: &impl ExpertExecutor,
    ) -> Result<DeepSeekV4LayerStepOutput> {
        self.decode_step_with_backend(
            state,
            hc_state,
            token_id,
            position,
            predicted_experts,
            expert_reader,
            expert_executor,
            DeepSeekV4OperatorBackend::Cpu,
        )
    }

    pub fn decode_step_with_backend(
        &self,
        state: &mut DeepSeekV4LayerState,
        hc_state: &[f32],
        token_id: u32,
        position: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        expert_executor: &impl ExpertExecutor,
        operator_backend: DeepSeekV4OperatorBackend,
    ) -> Result<DeepSeekV4LayerStepOutput> {
        let mut operators = DeepSeekV4OperatorContext::new(operator_backend)?;
        self.decode_step_with_operators(
            state,
            hc_state,
            token_id,
            position,
            predicted_experts,
            expert_reader,
            expert_executor,
            &mut operators,
        )
    }

    pub fn decode_step_with_operators(
        &self,
        state: &mut DeepSeekV4LayerState,
        hc_state: &[f32],
        token_id: u32,
        position: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        expert_executor: &impl ExpertExecutor,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<DeepSeekV4LayerStepOutput> {
        if hc_state.len() != self.hc_config.hc_hidden_size() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} HC input length mismatch: expected {}, got {}",
                self.layer,
                self.hc_config.hc_hidden_size(),
                hc_state.len()
            )));
        }

        let attention_pre = operators.hc_pre(hc_state, 1, self.hc_config, &self.hc_attention)?;
        let attention_input = operators.rms_norm(
            &attention_pre.hidden,
            &self.attn_norm,
            self.hc_config.norm_eps,
            "attn_norm",
        )?;
        let attention_hidden = self.attention.decode_step_with_operators(
            &mut state.kv,
            &attention_input,
            position,
            operators,
        )?;
        let after_attention = operators.hc_post(
            &attention_hidden,
            hc_state,
            self.hc_config,
            &attention_pre.split,
        )?;

        let ffn_pre =
            operators.hc_pre(&after_attention, 1, self.hc_config, &self.hc_feed_forward)?;
        let ffn_input = operators.rms_norm(
            &ffn_pre.hidden,
            &self.ffn_norm,
            self.hc_config.norm_eps,
            "ffn_norm",
        )?;
        let moe = operators.routed_moe_step(
            self.layer,
            &ffn_input,
            token_id,
            &self.router,
            predicted_experts,
            &self.router_policy,
            &mut state.expert_planner,
            expert_reader,
            &mut state.expert_handles,
            expert_executor,
            Some(&self.shared_ffn),
        )?;
        let hc_state = operators.hc_post(
            &moe.output,
            &after_attention,
            self.hc_config,
            &ffn_pre.split,
        )?;

        Ok(DeepSeekV4LayerStepOutput {
            attention_hidden,
            feed_forward_hidden: moe.output.clone(),
            moe,
            hc_state,
        })
    }

    pub fn prefill_start_with_operators(
        &self,
        state: &mut DeepSeekV4LayerState,
        hc_state: &[f32],
        token_ids: &[u32],
        start_pos: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        expert_executor: &impl ExpertExecutor,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        let tokens = token_ids.len();
        if tokens == 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} prefill requires at least one token",
                self.layer
            )));
        }
        if start_pos != 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} batched prefill currently supports start_pos=0, got {start_pos}",
                self.layer
            )));
        }
        if hc_state.len() != tokens * self.hc_config.hc_hidden_size() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} batched HC input length mismatch: expected {}, got {}",
                self.layer,
                tokens * self.hc_config.hc_hidden_size(),
                hc_state.len()
            )));
        }

        let attention_pre =
            operators.hc_pre(hc_state, tokens, self.hc_config, &self.hc_attention)?;
        let attention_input = rms_norm_rows_with_operators(
            operators,
            &attention_pre.hidden,
            tokens,
            &self.attn_norm,
            self.hc_config.norm_eps,
            "attn_norm",
        )?;
        let attention_hidden = self.attention.prefill_start_with_operators(
            &mut state.kv,
            &attention_input,
            start_pos,
            operators,
        )?;
        let after_attention = operators.hc_post(
            &attention_hidden,
            hc_state,
            self.hc_config,
            &attention_pre.split,
        )?;

        let ffn_pre = operators.hc_pre(
            &after_attention,
            tokens,
            self.hc_config,
            &self.hc_feed_forward,
        )?;
        let ffn_input = rms_norm_rows_with_operators(
            operators,
            &ffn_pre.hidden,
            tokens,
            &self.ffn_norm,
            self.hc_config.norm_eps,
            "ffn_norm",
        )?;
        let mut feed_forward_hidden = Vec::with_capacity(tokens * self.hc_config.hidden_size);
        for token in 0..tokens {
            let row = &ffn_input
                [token * self.hc_config.hidden_size..(token + 1) * self.hc_config.hidden_size];
            let moe = operators.routed_moe_step(
                self.layer,
                row,
                token_ids[token],
                &self.router,
                predicted_experts,
                &self.router_policy,
                &mut state.expert_planner,
                expert_reader,
                &mut state.expert_handles,
                expert_executor,
                Some(&self.shared_ffn),
            )?;
            feed_forward_hidden.extend_from_slice(&moe.output);
        }
        operators.hc_post(
            &feed_forward_hidden,
            &after_attention,
            self.hc_config,
            &ffn_pre.split,
        )
    }
}

#[derive(Debug, Clone)]
pub struct DeepSeekV4LayerState {
    pub kv: DeepSeekV4AttentionCache,
    pub expert_planner: ExpertStreamingPlanner,
    pub expert_handles: CpuExpertHandleStore,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4LayerStepOutput {
    pub attention_hidden: Vec<f32>,
    pub feed_forward_hidden: Vec<f32>,
    pub moe: RoutedMoeStepOutput,
    pub hc_state: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepSeekV4AttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub compress_ratio: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub original_seq_len: usize,
    pub rope_factor: f32,
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
}

impl DeepSeekV4AttentionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.num_heads == 0
            || self.head_dim == 0
            || self.o_groups == 0
            || self.o_lora_rank == 0
            || self.window_size == 0
            || self.index_n_heads == 0
            || self.index_head_dim == 0
            || self.index_topk == 0
        {
            return Err(Error::Model(format!(
                "invalid DeepSeek-V4 attention shape: hidden={}, heads={}, head_dim={}, groups={}, o_rank={}, window={}, index_heads={}, index_dim={}, index_topk={}",
                self.hidden_size,
                self.num_heads,
                self.head_dim,
                self.o_groups,
                self.o_lora_rank,
                self.window_size,
                self.index_n_heads,
                self.index_head_dim,
                self.index_topk
            )));
        }
        if self.num_heads % self.o_groups != 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 attention heads {} must be divisible by o_groups {}",
                self.num_heads, self.o_groups
            )));
        }
        if self.rope_head_dim > self.head_dim || self.rope_head_dim % 2 != 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 rope_head_dim {} must be even and <= head_dim {}",
                self.rope_head_dim, self.head_dim
            )));
        }
        if self.norm_eps <= 0.0 || self.rope_theta <= 0.0 || self.compress_rope_theta <= 0.0 {
            return Err(Error::Model(
                "DeepSeek-V4 attention eps/theta values must be positive".into(),
            ));
        }
        Ok(())
    }

    pub fn q_full_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn output_group_input_dim(&self) -> usize {
        self.q_full_dim() / self.o_groups
    }

    pub fn output_latent_dim(&self) -> usize {
        self.o_groups * self.o_lora_rank
    }

    pub fn sparse_spec(&self) -> SparseAttentionSpec {
        self.sparse_spec_with_topk(self.window_size)
    }

    pub fn sparse_spec_with_topk(&self, topk: usize) -> SparseAttentionSpec {
        SparseAttentionSpec {
            heads: self.num_heads,
            head_dim: self.head_dim,
            topk,
            softmax_scale: (self.head_dim as f32).powf(-0.5),
            has_attention_sink: true,
        }
    }

    pub fn rope_params(&self) -> DeepSeekV4RopeParams {
        if self.compress_ratio == 0 {
            DeepSeekV4RopeParams::plain(self.rope_theta)
        } else {
            DeepSeekV4RopeParams {
                theta: self.compress_rope_theta,
                original_seq_len: self.original_seq_len,
                factor: self.rope_factor,
                beta_fast: self.beta_fast,
                beta_slow: self.beta_slow,
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepSeekV4RopeParams {
    pub theta: f32,
    pub original_seq_len: usize,
    pub factor: f32,
    pub beta_fast: usize,
    pub beta_slow: usize,
}

impl DeepSeekV4RopeParams {
    pub fn plain(theta: f32) -> Self {
        Self {
            theta,
            original_seq_len: 0,
            factor: 1.0,
            beta_fast: 32,
            beta_slow: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4CompressorPayload {
    pub compress_ratio: usize,
    pub head_dim: usize,
    pub overlap: bool,
    pub rotate_for_indexer: bool,
    pub ape: Vec<f32>,
    pub ape_rows: usize,
    pub ape_cols: usize,
    pub norm: Vec<f32>,
    pub wkv: SourceLinearPayload,
    pub wgate: SourceLinearPayload,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4IndexerPayload {
    pub compressor: DeepSeekV4CompressorPayload,
    pub wq_b: SourceLinearPayload,
    pub weights_proj: SourceLinearPayload,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4CompressedAttentionPayload {
    pub compressor: DeepSeekV4CompressorPayload,
    pub indexer: Option<DeepSeekV4IndexerPayload>,
}

impl DeepSeekV4CompressedAttentionPayload {
    pub fn bind_optional(
        layer: usize,
        cfg: DeepSeekV4AttentionConfig,
        auxiliary: &[SourceTensorSlice],
        reader: &SourceTensorReader,
    ) -> Result<Option<Self>> {
        if cfg.compress_ratio == 0 {
            if !auxiliary.is_empty() {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 layer {layer} has non-compressed config but {} compressed attention auxiliary tensors",
                    auxiliary.len()
                )));
            }
            return Ok(None);
        }
        let compressor = DeepSeekV4CompressorPayload::bind(
            layer,
            auxiliary,
            reader,
            &format!("layers.{layer}.attn.compressor"),
            TensorRole::AttentionCompressor,
            cfg.compress_ratio,
            cfg.hidden_size,
            cfg.head_dim,
            false,
        )?;
        let indexer = if cfg.compress_ratio == 4 {
            Some(DeepSeekV4IndexerPayload::bind(
                layer, cfg, auxiliary, reader,
            )?)
        } else {
            None
        };
        Ok(Some(Self {
            compressor,
            indexer,
        }))
    }
}

impl DeepSeekV4CompressorPayload {
    #[allow(clippy::too_many_arguments)]
    fn bind(
        layer: usize,
        auxiliary: &[SourceTensorSlice],
        reader: &SourceTensorReader,
        prefix: &str,
        role: TensorRole,
        compress_ratio: usize,
        hidden_size: usize,
        head_dim: usize,
        rotate_for_indexer: bool,
    ) -> Result<Self> {
        let overlap = compress_ratio == 4;
        let coeff = if overlap { 2 } else { 1 };
        let out_dim = coeff * head_dim;
        let ape = read_aux_tensor_f32(auxiliary, reader, &format!("{prefix}.ape"))?;
        let ape_shape = two_dim_shape_from_payload(&ape, "compressor ape")?;
        let norm = decode_vector_f32(&read_aux_tensor(
            auxiliary,
            reader,
            &format!("{prefix}.norm.weight"),
        )?)?;
        let wkv = bind_aux_linear(
            auxiliary,
            reader,
            role.clone(),
            &format!("{prefix}.wkv.weight"),
            None,
        )?;
        let wgate = bind_aux_linear(
            auxiliary,
            reader,
            role,
            &format!("{prefix}.wgate.weight"),
            None,
        )?;

        if ape_shape != (compress_ratio, out_dim) {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} compressor '{prefix}.ape' shape mismatch: got {:?}, expected [{compress_ratio}, {out_dim}]",
                ape_shape
            )));
        }
        check_len(layer, &format!("{prefix}.norm"), norm.len(), head_dim)?;
        check_linear(layer, &format!("{prefix}.wkv"), &wkv, out_dim, hidden_size)?;
        check_linear(
            layer,
            &format!("{prefix}.wgate"),
            &wgate,
            out_dim,
            hidden_size,
        )?;
        Ok(Self {
            compress_ratio,
            head_dim,
            overlap,
            rotate_for_indexer,
            ape: decode_tensor_f32(&ape)?,
            ape_rows: ape_shape.0,
            ape_cols: ape_shape.1,
            norm,
            wkv,
            wgate,
        })
    }
}

impl DeepSeekV4IndexerPayload {
    fn bind(
        layer: usize,
        cfg: DeepSeekV4AttentionConfig,
        auxiliary: &[SourceTensorSlice],
        reader: &SourceTensorReader,
    ) -> Result<Self> {
        let prefix = format!("layers.{layer}.attn.indexer");
        let compressor = DeepSeekV4CompressorPayload::bind(
            layer,
            auxiliary,
            reader,
            &format!("{prefix}.compressor"),
            TensorRole::AuxIndexer,
            cfg.compress_ratio,
            cfg.hidden_size,
            cfg.index_head_dim,
            true,
        )?;
        let wq_b = bind_aux_linear(
            auxiliary,
            reader,
            TensorRole::AuxIndexer,
            &format!("{prefix}.wq_b.weight"),
            Some(&format!("{prefix}.wq_b.scale")),
        )?;
        let weights_proj = bind_aux_linear(
            auxiliary,
            reader,
            TensorRole::AuxIndexer,
            &format!("{prefix}.weights_proj.weight"),
            None,
        )?;
        check_linear(
            layer,
            "indexer.wq_b",
            &wq_b,
            cfg.index_n_heads * cfg.index_head_dim,
            cfg.q_lora_rank,
        )?;
        check_linear(
            layer,
            "indexer.weights_proj",
            &weights_proj,
            cfg.index_n_heads,
            cfg.hidden_size,
        )?;
        Ok(Self {
            compressor,
            wq_b,
            weights_proj,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4Attention {
    pub layer: usize,
    pub config: DeepSeekV4AttentionConfig,
    pub payload: AttentionSourcePayload,
    pub compressed: Option<DeepSeekV4CompressedAttentionPayload>,
}

impl DeepSeekV4Attention {
    pub fn new(
        layer: usize,
        config: DeepSeekV4AttentionConfig,
        payload: AttentionSourcePayload,
    ) -> Result<Self> {
        Self::new_with_compressed(layer, config, payload, None)
    }

    pub fn new_with_compressed(
        layer: usize,
        config: DeepSeekV4AttentionConfig,
        payload: AttentionSourcePayload,
        compressed: Option<DeepSeekV4CompressedAttentionPayload>,
    ) -> Result<Self> {
        config.validate()?;
        if config.compress_ratio == 0 && compressed.is_some() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} has non-compressed attention but compressed payload is present"
            )));
        }
        let attention = Self {
            layer,
            config,
            payload,
            compressed,
        };
        attention.validate_shapes()?;
        Ok(attention)
    }

    pub fn validate_shapes(&self) -> Result<()> {
        let cfg = self.config;
        check_linear(
            self.layer,
            "wq_a",
            &self.payload.query_a,
            cfg.q_lora_rank,
            cfg.hidden_size,
        )?;
        check_linear(
            self.layer,
            "wq_b",
            &self.payload.query_b,
            cfg.q_full_dim(),
            cfg.q_lora_rank,
        )?;
        check_linear(
            self.layer,
            "wkv",
            &self.payload.key_value,
            cfg.head_dim,
            cfg.hidden_size,
        )?;
        check_linear(
            self.layer,
            "wo_a",
            &self.payload.output_a,
            cfg.output_latent_dim(),
            cfg.output_group_input_dim(),
        )?;
        check_linear(
            self.layer,
            "wo_b",
            &self.payload.output_b,
            cfg.hidden_size,
            cfg.output_latent_dim(),
        )?;
        check_len(
            self.layer,
            "q_norm",
            self.payload.query_norm.len(),
            cfg.q_lora_rank,
        )?;
        check_len(
            self.layer,
            "kv_norm",
            self.payload.key_value_norm.len(),
            cfg.head_dim,
        )?;
        check_len(
            self.layer,
            "attn_sink",
            self.payload.attention_sink.len(),
            cfg.num_heads,
        )?;
        Ok(())
    }

    pub fn prefill_start_with_operators(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden: &[f32],
        start_pos: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        let cfg = self.config;
        if start_pos != 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} attention batched prefill currently supports start_pos=0, got {start_pos}",
                self.layer
            )));
        }
        if hidden.is_empty() || hidden.len() % cfg.hidden_size != 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} prefill hidden length mismatch: hidden={} dim={}",
                self.layer,
                hidden.len(),
                cfg.hidden_size
            )));
        }
        if cfg.compress_ratio == 0 {
            self.prefill_start_no_compress_with_operators(cache, hidden, start_pos, operators)
        } else {
            self.prefill_start_compressed_with_operators(cache, hidden, start_pos, operators)
        }
    }

    fn prefill_start_no_compress_with_operators(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden: &[f32],
        start_pos: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        let cfg = self.config;
        let tokens = hidden.len() / cfg.hidden_size;
        if cache.window.head_dim != cfg.head_dim || cache.window.window_size != cfg.window_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} KV cache shape mismatch: cache window={} head_dim={}, expected window={} head_dim={}",
                self.layer, cache.window.window_size, cache.window.head_dim, cfg.window_size, cfg.head_dim
            )));
        }

        let mut queries = Vec::with_capacity(tokens * cfg.q_full_dim());
        let mut values = Vec::with_capacity(tokens * cfg.head_dim);
        for token in 0..tokens {
            let position = start_pos + token;
            let row = &hidden[token * cfg.hidden_size..(token + 1) * cfg.hidden_size];
            let q_latent = operators.linear_matvec(&self.payload.query_a, row)?;
            let q_latent =
                operators.rms_norm(&q_latent, &self.payload.query_norm, cfg.norm_eps, "q_norm")?;
            let mut query = operators.linear_matvec(&self.payload.query_b, &q_latent)?;
            operators.rms_norm_heads_in_place(
                &mut query,
                cfg.num_heads,
                cfg.head_dim,
                cfg.norm_eps,
                self.layer,
            )?;
            apply_rotary_tail_scaled(
                &mut query,
                cfg.num_heads,
                cfg.head_dim,
                cfg.rope_head_dim,
                position,
                cfg.rope_params(),
                false,
            )?;
            queries.extend_from_slice(&query);

            let kv = operators.linear_matvec(&self.payload.key_value, row)?;
            let mut kv =
                operators.rms_norm(&kv, &self.payload.key_value_norm, cfg.norm_eps, "kv_norm")?;
            apply_rotary_tail_scaled(
                &mut kv,
                1,
                cfg.head_dim,
                cfg.rope_head_dim,
                position,
                cfg.rope_params(),
                false,
            )?;
            cache.window.append(position, &kv)?;
            values.extend_from_slice(&kv);
        }

        let topk_cols = tokens.min(cfg.window_size);
        let topk = window_topk_indices_prefill(cfg.window_size, tokens);
        let context = operators.sparse_attention(
            &queries,
            &values,
            &topk,
            &self.payload.attention_sink,
            tokens,
            tokens,
            cfg.sparse_spec_with_topk(topk_cols),
        )?;
        self.project_context_rows_with_operators(context, start_pos, tokens, operators)
    }

    fn prefill_start_compressed_with_operators(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden: &[f32],
        start_pos: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        let cfg = self.config;
        let tokens = hidden.len() / cfg.hidden_size;
        if cache.window.head_dim != cfg.head_dim || cache.window.window_size != cfg.window_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} KV cache shape mismatch: cache window={} head_dim={}, expected window={} head_dim={}",
                self.layer, cache.window.window_size, cache.window.head_dim, cfg.window_size, cfg.head_dim
            )));
        }
        let compressed = self.compressed.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {} has compress_ratio {} but no typed compressed attention payload is bound",
                self.layer, cfg.compress_ratio
            ))
        })?;
        let rope = cfg.rope_params();

        let mut queries = Vec::with_capacity(tokens * cfg.q_full_dim());
        let mut q_latents = Vec::with_capacity(tokens * cfg.q_lora_rank);
        let mut window_values = Vec::with_capacity(tokens * cfg.head_dim);
        for token in 0..tokens {
            let position = start_pos + token;
            let row = &hidden[token * cfg.hidden_size..(token + 1) * cfg.hidden_size];
            let q_latent = operators.linear_matvec(&self.payload.query_a, row)?;
            let q_latent =
                operators.rms_norm(&q_latent, &self.payload.query_norm, cfg.norm_eps, "q_norm")?;
            let mut query = operators.linear_matvec(&self.payload.query_b, &q_latent)?;
            operators.rms_norm_heads_in_place(
                &mut query,
                cfg.num_heads,
                cfg.head_dim,
                cfg.norm_eps,
                self.layer,
            )?;
            apply_rotary_tail_scaled(
                &mut query,
                cfg.num_heads,
                cfg.head_dim,
                cfg.rope_head_dim,
                position,
                rope,
                false,
            )?;
            q_latents.extend_from_slice(&q_latent);
            queries.extend_from_slice(&query);

            let kv = operators.linear_matvec(&self.payload.key_value, row)?;
            let mut kv =
                operators.rms_norm(&kv, &self.payload.key_value_norm, cfg.norm_eps, "kv_norm")?;
            apply_rotary_tail_scaled(
                &mut kv,
                1,
                cfg.head_dim,
                cfg.rope_head_dim,
                position,
                rope,
                false,
            )?;
            cache.window.append(position, &kv)?;
            window_values.extend_from_slice(&kv);
        }

        if let Some(indexer) = compressed.indexer.as_ref() {
            let indexer_values = {
                let state = cache.indexer_compressor.as_mut().ok_or_else(|| {
                    Error::Model(format!(
                        "DeepSeek-V4 layer {} missing indexer compressor state",
                        self.layer
                    ))
                })?;
                state.prefill_start(
                    &indexer.compressor,
                    hidden,
                    cfg.rope_head_dim,
                    rope,
                    operators,
                )?
            };
            if indexer_values.len() % cfg.index_head_dim != 0 {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 layer {} indexer prefill compressed length {} is not divisible by {}",
                    self.layer,
                    indexer_values.len(),
                    cfg.index_head_dim
                )));
            }
            cache.indexer_compressed.extend_from_slice(&indexer_values);
        }

        let main_compressed = {
            let state = cache.main_compressor.as_mut().ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 layer {} missing main compressor state",
                    self.layer
                ))
            })?;
            state.prefill_start(
                &compressed.compressor,
                hidden,
                cfg.rope_head_dim,
                rope,
                operators,
            )?
        };
        if main_compressed.len() % cfg.head_dim != 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} main prefill compressed length {} is not divisible by {}",
                self.layer,
                main_compressed.len(),
                cfg.head_dim
            )));
        }
        cache.compressed.extend_from_slice(&main_compressed);

        let window_cols = tokens.min(cfg.window_size);
        let mut topk = window_topk_indices_prefill(cfg.window_size, tokens);
        let mut topk_cols = window_cols;
        let compressed_offset = tokens;
        if let Some(indexer) = compressed.indexer.as_ref() {
            let (indexer_topk, indexer_cols) = indexer_topk_indices_prefill(
                indexer,
                cfg,
                &q_latents,
                hidden,
                &cache.indexer_compressed,
                compressed_offset,
                operators,
            )?;
            if indexer_cols > 0 {
                topk = concat_topk_rows(&topk, topk_cols, &indexer_topk, indexer_cols, tokens)?;
                topk_cols += indexer_cols;
            }
        } else {
            let (compressed_topk, compressed_cols) =
                compress_topk_indices_prefill(cfg.compress_ratio, tokens, compressed_offset);
            if compressed_cols > 0 {
                topk =
                    concat_topk_rows(&topk, topk_cols, &compressed_topk, compressed_cols, tokens)?;
                topk_cols += compressed_cols;
            }
        }

        let main_compressed_len = main_compressed.len() / cfg.head_dim;
        let mut values = Vec::with_capacity((tokens + main_compressed_len) * cfg.head_dim);
        values.extend_from_slice(&window_values);
        values.extend_from_slice(&main_compressed);
        let context = operators.sparse_attention(
            &queries,
            &values,
            &topk,
            &self.payload.attention_sink,
            tokens,
            tokens + main_compressed_len,
            cfg.sparse_spec_with_topk(topk_cols),
        )?;
        self.project_context_rows_with_operators(context, start_pos, tokens, operators)
    }

    fn project_context_rows_with_operators(
        &self,
        mut context: Vec<f32>,
        start_pos: usize,
        tokens: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        let cfg = self.config;
        if context.len() != tokens * cfg.q_full_dim() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} attention context length mismatch: expected {}, got {}",
                self.layer,
                tokens * cfg.q_full_dim(),
                context.len()
            )));
        }
        let mut out = Vec::with_capacity(tokens * cfg.hidden_size);
        for token in 0..tokens {
            let position = start_pos + token;
            let row_start = token * cfg.q_full_dim();
            let row = &mut context[row_start..row_start + cfg.q_full_dim()];
            apply_rotary_tail_scaled(
                row,
                cfg.num_heads,
                cfg.head_dim,
                cfg.rope_head_dim,
                position,
                cfg.rope_params(),
                true,
            )?;
            let latent =
                operators.grouped_output_a(&self.payload.output_a, row, cfg, self.layer)?;
            let projected = operators.linear_matvec(&self.payload.output_b, &latent)?;
            out.extend_from_slice(&projected);
        }
        Ok(out)
    }

    pub fn decode_step_reference(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden: &[f32],
        position: usize,
    ) -> Result<Vec<f32>> {
        self.decode_step_with_backend(cache, hidden, position, DeepSeekV4OperatorBackend::Cpu)
    }

    pub fn decode_step_with_backend(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden: &[f32],
        position: usize,
        operator_backend: DeepSeekV4OperatorBackend,
    ) -> Result<Vec<f32>> {
        let mut operators = DeepSeekV4OperatorContext::new(operator_backend)?;
        self.decode_step_with_operators(cache, hidden, position, &mut operators)
    }

    pub fn decode_step_with_operators(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden: &[f32],
        position: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        let cfg = self.config;
        if cfg.compress_ratio == 0 {
            return self.decode_step_no_compress_with_operators(
                &mut cache.window,
                hidden,
                position,
                operators,
            );
        }
        if hidden.len() != cfg.hidden_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} attention input mismatch: expected {}, got {}",
                self.layer,
                cfg.hidden_size,
                hidden.len()
            )));
        }
        if cache.window.head_dim != cfg.head_dim || cache.window.window_size != cfg.window_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} KV cache shape mismatch: cache window={} head_dim={}, expected window={} head_dim={}",
                self.layer, cache.window.window_size, cache.window.head_dim, cfg.window_size, cfg.head_dim
            )));
        }
        let compressed = self.compressed.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {} has compress_ratio {} but no typed compressed attention payload is bound",
                self.layer, cfg.compress_ratio
            ))
        })?;
        let rope = cfg.rope_params();

        let q_latent = operators.linear_matvec(&self.payload.query_a, hidden)?;
        let q_latent =
            operators.rms_norm(&q_latent, &self.payload.query_norm, cfg.norm_eps, "q_norm")?;
        let mut query = operators.linear_matvec(&self.payload.query_b, &q_latent)?;
        operators.rms_norm_heads_in_place(
            &mut query,
            cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
            self.layer,
        )?;
        apply_rotary_tail_scaled(
            &mut query,
            cfg.num_heads,
            cfg.head_dim,
            cfg.rope_head_dim,
            position,
            rope,
            false,
        )?;

        let kv = operators.linear_matvec(&self.payload.key_value, hidden)?;
        let mut kv =
            operators.rms_norm(&kv, &self.payload.key_value_norm, cfg.norm_eps, "kv_norm")?;
        apply_rotary_tail_scaled(
            &mut kv,
            1,
            cfg.head_dim,
            cfg.rope_head_dim,
            position,
            rope,
            false,
        )?;
        cache.window.append(position, &kv)?;

        if let Some(indexer) = compressed.indexer.as_ref() {
            let new_indexer_kv = {
                let state = cache.indexer_compressor.as_mut().ok_or_else(|| {
                    Error::Model(format!(
                        "DeepSeek-V4 layer {} missing indexer compressor state",
                        self.layer
                    ))
                })?;
                state.append_step(
                    &indexer.compressor,
                    hidden,
                    position,
                    cfg.rope_head_dim,
                    rope,
                    operators,
                )?
            };
            if let Some(value) = new_indexer_kv {
                cache.append_indexer_compressed(&value, cfg.index_head_dim)?;
            }
        }

        let new_main_kv = {
            let state = cache.main_compressor.as_mut().ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 layer {} missing main compressor state",
                    self.layer
                ))
            })?;
            state.append_step(
                &compressed.compressor,
                hidden,
                position,
                cfg.rope_head_dim,
                rope,
                operators,
            )?
        };
        if let Some(value) = new_main_kv {
            cache.append_compressed(&value)?;
        }

        let mut topk = cache.window.topk_indices(position, cfg.window_size);
        if let Some(indexer) = compressed.indexer.as_ref() {
            topk.extend(indexer_topk_indices(
                indexer,
                cfg,
                &q_latent,
                hidden,
                position,
                &cache.indexer_compressed,
                cfg.window_size,
                operators,
            )?);
        } else {
            topk.extend((0..cache.compressed_len()).map(|idx| (cfg.window_size + idx) as isize));
        }

        let values = cache.combined_values_for_attention();
        let mut context = operators.sparse_attention(
            &query,
            &values,
            &topk,
            &self.payload.attention_sink,
            1,
            cache.kv_len_for_attention(),
            cfg.sparse_spec_with_topk(topk.len()),
        )?;
        apply_rotary_tail_scaled(
            &mut context,
            cfg.num_heads,
            cfg.head_dim,
            cfg.rope_head_dim,
            position,
            rope,
            true,
        )?;
        let latent =
            operators.grouped_output_a(&self.payload.output_a, &context, cfg, self.layer)?;
        operators.linear_matvec(&self.payload.output_b, &latent)
    }

    /// Execute the official DSV4 attention path for non-compressed layers
    /// (`compress_ratio == 0`, i.e. local sliding-window MLA only).
    pub fn decode_step_no_compress(
        &self,
        cache: &mut DeepSeekV4WindowKvCache,
        hidden: &[f32],
        position: usize,
    ) -> Result<Vec<f32>> {
        self.decode_step_no_compress_with_backend(
            cache,
            hidden,
            position,
            DeepSeekV4OperatorBackend::Cpu,
        )
    }

    pub fn decode_step_no_compress_with_backend(
        &self,
        cache: &mut DeepSeekV4WindowKvCache,
        hidden: &[f32],
        position: usize,
        operator_backend: DeepSeekV4OperatorBackend,
    ) -> Result<Vec<f32>> {
        let mut operators = DeepSeekV4OperatorContext::new(operator_backend)?;
        self.decode_step_no_compress_with_operators(cache, hidden, position, &mut operators)
    }

    pub fn decode_step_no_compress_with_operators(
        &self,
        cache: &mut DeepSeekV4WindowKvCache,
        hidden: &[f32],
        position: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        let cfg = self.config;
        if cfg.compress_ratio != 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} has compress_ratio {}; compressed CSA/HCA attention is not implemented in this path",
                self.layer, cfg.compress_ratio
            )));
        }
        if hidden.len() != cfg.hidden_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} attention input mismatch: expected {}, got {}",
                self.layer,
                cfg.hidden_size,
                hidden.len()
            )));
        }
        if cache.head_dim != cfg.head_dim || cache.window_size != cfg.window_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} KV cache shape mismatch: cache window={} head_dim={}, expected window={} head_dim={}",
                self.layer, cache.window_size, cache.head_dim, cfg.window_size, cfg.head_dim
            )));
        }

        let q_latent = operators.linear_matvec(&self.payload.query_a, hidden)?;
        let q_latent =
            operators.rms_norm(&q_latent, &self.payload.query_norm, cfg.norm_eps, "q_norm")?;
        let mut query = operators.linear_matvec(&self.payload.query_b, &q_latent)?;
        operators.rms_norm_heads_in_place(
            &mut query,
            cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
            self.layer,
        )?;
        apply_rotary_tail(
            &mut query,
            cfg.num_heads,
            cfg.head_dim,
            cfg.rope_head_dim,
            position,
            cfg.rope_theta,
            false,
        )?;

        let kv = operators.linear_matvec(&self.payload.key_value, hidden)?;
        let mut kv =
            operators.rms_norm(&kv, &self.payload.key_value_norm, cfg.norm_eps, "kv_norm")?;
        apply_rotary_tail(
            &mut kv,
            1,
            cfg.head_dim,
            cfg.rope_head_dim,
            position,
            cfg.rope_theta,
            false,
        )?;
        cache.append(position, &kv)?;
        let topk = cache.topk_indices(position, cfg.window_size);
        let values = cache.values_for_attention();
        let mut context = operators.sparse_attention(
            &query,
            values,
            &topk,
            &self.payload.attention_sink,
            1,
            cache.kv_len_for_attention(),
            cfg.sparse_spec(),
        )?;
        apply_rotary_tail(
            &mut context,
            cfg.num_heads,
            cfg.head_dim,
            cfg.rope_head_dim,
            position,
            cfg.rope_theta,
            true,
        )?;
        let latent =
            operators.grouped_output_a(&self.payload.output_a, &context, cfg, self.layer)?;
        operators.linear_matvec(&self.payload.output_b, &latent)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4AttentionCache {
    pub window: DeepSeekV4WindowKvCache,
    pub compressed: Vec<f32>,
    pub indexer_compressed: Vec<f32>,
    main_compressor: Option<DeepSeekV4CompressorState>,
    indexer_compressor: Option<DeepSeekV4CompressorState>,
}

impl DeepSeekV4AttentionCache {
    pub fn new(cfg: DeepSeekV4AttentionConfig) -> Self {
        let compressed = cfg.compress_ratio != 0;
        Self {
            window: DeepSeekV4WindowKvCache::new(cfg.window_size, cfg.head_dim),
            compressed: Vec::new(),
            indexer_compressed: Vec::new(),
            main_compressor: compressed.then(|| {
                DeepSeekV4CompressorState::new(
                    cfg.compress_ratio,
                    cfg.head_dim,
                    cfg.compress_ratio == 4,
                )
            }),
            indexer_compressor: (cfg.compress_ratio == 4).then(|| {
                DeepSeekV4CompressorState::new(cfg.compress_ratio, cfg.index_head_dim, true)
            }),
        }
    }

    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn compressed_len(&self) -> usize {
        self.compressed.len() / self.window.head_dim
    }

    pub fn indexer_compressed_len(&self, index_head_dim: usize) -> usize {
        if index_head_dim == 0 {
            0
        } else {
            self.indexer_compressed.len() / index_head_dim
        }
    }

    fn append_compressed(&mut self, value: &[f32]) -> Result<()> {
        if value.len() != self.window.head_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 compressed KV append mismatch: expected {}, got {}",
                self.window.head_dim,
                value.len()
            )));
        }
        self.compressed.extend_from_slice(value);
        Ok(())
    }

    fn append_indexer_compressed(&mut self, value: &[f32], index_head_dim: usize) -> Result<()> {
        if value.len() != index_head_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 indexer compressed KV append mismatch: expected {index_head_dim}, got {}",
                value.len()
            )));
        }
        self.indexer_compressed.extend_from_slice(value);
        Ok(())
    }

    fn combined_values_for_attention(&self) -> Vec<f32> {
        let mut values =
            Vec::with_capacity(self.window.values_full().len() + self.compressed.len());
        values.extend_from_slice(self.window.values_full());
        values.extend_from_slice(&self.compressed);
        values
    }

    fn kv_len_for_attention(&self) -> usize {
        self.window.window_size + self.compressed_len()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct DeepSeekV4CompressorState {
    ratio: usize,
    head_dim: usize,
    out_dim: usize,
    overlap: bool,
    kv_state: Vec<f32>,
    score_state: Vec<f32>,
}

impl DeepSeekV4CompressorState {
    fn new(ratio: usize, head_dim: usize, overlap: bool) -> Self {
        let coeff = if overlap { 2 } else { 1 };
        let out_dim = coeff * head_dim;
        Self {
            ratio,
            head_dim,
            out_dim,
            overlap,
            kv_state: vec![0.0; coeff * ratio * out_dim],
            score_state: vec![f32::NEG_INFINITY; coeff * ratio * out_dim],
        }
    }

    fn prefill_start(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        hidden: &[f32],
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        if payload.compress_ratio != self.ratio
            || payload.head_dim != self.head_dim
            || payload.overlap != self.overlap
        {
            return Err(Error::Model(
                "DeepSeek-V4 compressor payload/state shape mismatch".into(),
            ));
        }
        if hidden.is_empty() || hidden.len() % payload.wkv.format.in_features() != 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 compressor prefill hidden length mismatch: hidden={} in_features={}",
                hidden.len(),
                payload.wkv.format.in_features()
            )));
        }
        self.kv_state.fill(0.0);
        self.score_state.fill(f32::NEG_INFINITY);

        let tokens = hidden.len() / payload.wkv.format.in_features();
        let mut kv_rows = Vec::with_capacity(tokens * self.out_dim);
        let mut score_rows = Vec::with_capacity(tokens * self.out_dim);
        for token in 0..tokens {
            let row = &hidden[token * payload.wkv.format.in_features()
                ..(token + 1) * payload.wkv.format.in_features()];
            let kv = operators.linear_matvec(&payload.wkv, row)?;
            let score = operators.linear_matvec(&payload.wgate, row)?;
            if kv.len() != self.out_dim || score.len() != self.out_dim {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 compressor prefill matvec length mismatch: kv={} score={} expected {}",
                    kv.len(),
                    score.len(),
                    self.out_dim
                )));
            }
            kv_rows.extend_from_slice(&kv);
            score_rows.extend_from_slice(&score);
        }

        let remainder = tokens % self.ratio;
        let cutoff = tokens - remainder;
        let groups = cutoff / self.ratio;
        let state_offset = if self.overlap { self.ratio } else { 0 };

        if self.overlap && cutoff >= self.ratio {
            for row in 0..self.ratio {
                let src_token = cutoff - self.ratio + row;
                let src = src_token * self.out_dim;
                let dst = row * self.out_dim;
                self.kv_state[dst..dst + self.out_dim]
                    .copy_from_slice(&kv_rows[src..src + self.out_dim]);
                for dim in 0..self.out_dim {
                    self.score_state[dst + dim] =
                        score_rows[src + dim] + payload.ape[row * payload.ape_cols + dim];
                }
            }
        }
        if remainder > 0 {
            for row in 0..remainder {
                let src_token = cutoff + row;
                let src = src_token * self.out_dim;
                let dst = (state_offset + row) * self.out_dim;
                self.kv_state[dst..dst + self.out_dim]
                    .copy_from_slice(&kv_rows[src..src + self.out_dim]);
                for dim in 0..self.out_dim {
                    self.score_state[dst + dim] =
                        score_rows[src + dim] + payload.ape[row * payload.ape_cols + dim];
                }
            }
        }

        if groups == 0 {
            return Ok(Vec::new());
        }

        let rows_per_group = if self.overlap {
            2 * self.ratio
        } else {
            self.ratio
        };
        let mut out_all = Vec::with_capacity(groups * self.head_dim);
        for group in 0..groups {
            let mut kv_group = vec![0.0f32; rows_per_group * self.head_dim];
            let mut score_group = vec![f32::NEG_INFINITY; rows_per_group * self.head_dim];
            if self.overlap {
                if group > 0 {
                    for row in 0..self.ratio {
                        let src_token = (group - 1) * self.ratio + row;
                        let src = src_token * self.out_dim;
                        let dst = row * self.head_dim;
                        kv_group[dst..dst + self.head_dim]
                            .copy_from_slice(&kv_rows[src..src + self.head_dim]);
                        for dim in 0..self.head_dim {
                            score_group[dst + dim] =
                                score_rows[src + dim] + payload.ape[row * payload.ape_cols + dim];
                        }
                    }
                }
                for row in 0..self.ratio {
                    let src_token = group * self.ratio + row;
                    let src = src_token * self.out_dim + self.head_dim;
                    let dst = (self.ratio + row) * self.head_dim;
                    kv_group[dst..dst + self.head_dim]
                        .copy_from_slice(&kv_rows[src..src + self.head_dim]);
                    for dim in 0..self.head_dim {
                        score_group[dst + dim] = score_rows[src + dim]
                            + payload.ape[row * payload.ape_cols + self.head_dim + dim];
                    }
                }
            } else {
                for row in 0..self.ratio {
                    let src_token = group * self.ratio + row;
                    let src = src_token * self.out_dim;
                    let dst = row * self.head_dim;
                    kv_group[dst..dst + self.head_dim]
                        .copy_from_slice(&kv_rows[src..src + self.head_dim]);
                    for dim in 0..self.head_dim {
                        score_group[dst + dim] =
                            score_rows[src + dim] + payload.ape[row * payload.ape_cols + dim];
                    }
                }
            }
            let mut out =
                compress_rows_softmax(&kv_group, &score_group, rows_per_group, self.head_dim)?;
            out = operators.rms_norm(&out, &payload.norm, 1e-6, "compressor.norm")?;
            apply_rotary_tail_scaled(
                &mut out,
                1,
                self.head_dim,
                rope_dim.min(self.head_dim),
                group * self.ratio,
                rope,
                false,
            )?;
            out_all.extend_from_slice(&out);
        }
        Ok(out_all)
    }

    fn append_step(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        hidden: &[f32],
        position: usize,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Option<Vec<f32>>> {
        if payload.compress_ratio != self.ratio
            || payload.head_dim != self.head_dim
            || payload.overlap != self.overlap
        {
            return Err(Error::Model(
                "DeepSeek-V4 compressor payload/state shape mismatch".into(),
            ));
        }
        let kv = operators.linear_matvec(&payload.wkv, hidden)?;
        let mut score = operators.linear_matvec(&payload.wgate, hidden)?;
        if kv.len() != self.out_dim || score.len() != self.out_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 compressor matvec length mismatch: kv={} score={} expected {}",
                kv.len(),
                score.len(),
                self.out_dim
            )));
        }
        let ape_row = position % self.ratio;
        let ape_start = ape_row * payload.ape_cols;
        for (value, ape) in score
            .iter_mut()
            .zip(&payload.ape[ape_start..ape_start + payload.ape_cols])
        {
            *value += *ape;
        }
        let row = if self.overlap {
            self.ratio + position % self.ratio
        } else {
            position % self.ratio
        };
        let dst = row * self.out_dim;
        self.kv_state[dst..dst + self.out_dim].copy_from_slice(&kv);
        self.score_state[dst..dst + self.out_dim].copy_from_slice(&score);

        if (position + 1) % self.ratio != 0 {
            return Ok(None);
        }

        let compressed = self.compress_current_window(
            payload,
            position + 1 - self.ratio,
            rope_dim,
            rope,
            operators,
        )?;
        if self.overlap {
            let split = self.ratio * self.out_dim;
            let (prev, current) = self.kv_state.split_at_mut(split);
            prev.copy_from_slice(current);
            let (prev, current) = self.score_state.split_at_mut(split);
            prev.copy_from_slice(current);
        }
        Ok(Some(compressed))
    }

    fn compress_current_window(
        &self,
        payload: &DeepSeekV4CompressorPayload,
        compressed_position: usize,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        let rows = if self.overlap {
            2 * self.ratio
        } else {
            self.ratio
        };
        let mut kv_rows = vec![0.0f32; rows * self.head_dim];
        let mut score_rows = vec![f32::NEG_INFINITY; rows * self.head_dim];
        if self.overlap {
            for row in 0..self.ratio {
                let src = row * self.out_dim;
                let dst = row * self.head_dim;
                kv_rows[dst..dst + self.head_dim]
                    .copy_from_slice(&self.kv_state[src..src + self.head_dim]);
                score_rows[dst..dst + self.head_dim]
                    .copy_from_slice(&self.score_state[src..src + self.head_dim]);
            }
            for row in 0..self.ratio {
                let src = (self.ratio + row) * self.out_dim + self.head_dim;
                let dst = (self.ratio + row) * self.head_dim;
                kv_rows[dst..dst + self.head_dim]
                    .copy_from_slice(&self.kv_state[src..src + self.head_dim]);
                score_rows[dst..dst + self.head_dim]
                    .copy_from_slice(&self.score_state[src..src + self.head_dim]);
            }
        } else {
            kv_rows.copy_from_slice(&self.kv_state);
            score_rows.copy_from_slice(&self.score_state);
        }

        let mut out = vec![0.0f32; self.head_dim];
        for dim in 0..self.head_dim {
            let mut max_score = f32::NEG_INFINITY;
            for row in 0..rows {
                max_score = max_score.max(score_rows[row * self.head_dim + dim]);
            }
            if !max_score.is_finite() {
                continue;
            }
            let mut denom = 0.0f32;
            for row in 0..rows {
                let score = score_rows[row * self.head_dim + dim];
                if score.is_finite() {
                    denom += (score - max_score).exp();
                }
            }
            if denom == 0.0 || !denom.is_finite() {
                return Err(Error::Model(
                    "DeepSeek-V4 compressor softmax denominator is invalid".into(),
                ));
            }
            for row in 0..rows {
                let score = score_rows[row * self.head_dim + dim];
                if score.is_finite() {
                    let weight = (score - max_score).exp() / denom;
                    out[dim] += weight * kv_rows[row * self.head_dim + dim];
                }
            }
        }
        let mut out = operators.rms_norm(&out, &payload.norm, 1e-6, "compressor.norm")?;
        apply_rotary_tail_scaled(
            &mut out,
            1,
            self.head_dim,
            rope_dim.min(self.head_dim),
            compressed_position,
            rope,
            false,
        )?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4WindowKvCache {
    window_size: usize,
    head_dim: usize,
    len: usize,
    values: Vec<f32>,
}

impl DeepSeekV4WindowKvCache {
    pub fn new(window_size: usize, head_dim: usize) -> Self {
        Self {
            window_size,
            head_dim,
            len: 0,
            values: vec![0.0; window_size * head_dim],
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn append(&mut self, position: usize, value: &[f32]) -> Result<()> {
        if value.len() != self.head_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 KV append mismatch: expected {}, got {}",
                self.head_dim,
                value.len()
            )));
        }
        let slot = position % self.window_size;
        let start = slot * self.head_dim;
        self.values[start..start + self.head_dim].copy_from_slice(value);
        self.len = self.window_size.min(self.len + 1);
        Ok(())
    }

    pub fn topk_indices(&self, position: usize, topk: usize) -> Vec<isize> {
        let mut out = vec![-1; topk];
        if self.len == 0 || topk == 0 {
            return out;
        }
        if self.len < self.window_size {
            let take = self.len.min(topk);
            for (slot, value) in out.iter_mut().take(take).enumerate() {
                *value = slot as isize;
            }
            return out;
        }
        let slot = position % self.window_size;
        let mut write = 0usize;
        for idx in slot + 1..self.window_size {
            if write < topk {
                out[write] = idx as isize;
                write += 1;
            }
        }
        for idx in 0..=slot {
            if write < topk {
                out[write] = idx as isize;
                write += 1;
            }
        }
        out
    }

    fn kv_len_for_attention(&self) -> usize {
        self.len
    }

    fn values_for_attention(&self) -> &[f32] {
        &self.values[..self.len * self.head_dim]
    }

    fn values_full(&self) -> &[f32] {
        &self.values
    }
}

fn bind_aux_linear(
    auxiliary: &[SourceTensorSlice],
    reader: &SourceTensorReader,
    role: TensorRole,
    weight_name: &str,
    scale_name: Option<&str>,
) -> Result<SourceLinearPayload> {
    let weight = read_aux_tensor(auxiliary, reader, weight_name)?;
    let scale = scale_name
        .map(|name| read_aux_tensor(auxiliary, reader, name))
        .transpose()?;
    SourceLinearPayload::from_weight_and_scale(role, weight, scale)
}

fn read_aux_tensor(
    auxiliary: &[SourceTensorSlice],
    reader: &SourceTensorReader,
    name: &str,
) -> Result<SourceTensorPayload> {
    let slice = auxiliary
        .iter()
        .find(|slice| slice.name == name)
        .ok_or_else(|| Error::Model(format!("DeepSeek-V4 missing auxiliary tensor '{name}'")))?;
    reader.read_slice(slice)
}

fn read_aux_tensor_f32(
    auxiliary: &[SourceTensorSlice],
    reader: &SourceTensorReader,
    name: &str,
) -> Result<SourceTensorPayload> {
    let payload = read_aux_tensor(auxiliary, reader, name)?;
    let _ = decode_tensor_f32(&payload)?;
    Ok(payload)
}

fn two_dim_shape_from_payload(
    payload: &SourceTensorPayload,
    label: &str,
) -> Result<(usize, usize)> {
    let [rows, cols]: [usize; 2] =
        payload
            .slice
            .shape
            .clone()
            .try_into()
            .map_err(|shape: Vec<usize>| {
                Error::Model(format!(
                    "DeepSeek-V4 {label} '{}' expects 2D shape, got {:?}",
                    payload.slice.name, shape
                ))
            })?;
    Ok((rows, cols))
}

fn check_linear(
    layer: usize,
    label: &str,
    linear: &SourceLinearPayload,
    out: usize,
    input: usize,
) -> Result<()> {
    if linear.format.out_features() != out || linear.format.in_features() != input {
        return Err(Error::Model(format!(
            "DeepSeek-V4 layer {layer} {label} shape mismatch: got [{}, {}], expected [{out}, {input}]",
            linear.format.out_features(),
            linear.format.in_features()
        )));
    }
    Ok(())
}

fn check_len(layer: usize, label: &str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(Error::Model(format!(
            "DeepSeek-V4 layer {layer} {label} length mismatch: got {got}, expected {expected}"
        )));
    }
    Ok(())
}

fn rms_norm_rows_with_operators(
    operators: &mut DeepSeekV4OperatorContext,
    input: &[f32],
    tokens: usize,
    weight: &[f32],
    eps: f32,
    label: &str,
) -> Result<Vec<f32>> {
    if tokens == 0 || weight.is_empty() || input.len() != tokens * weight.len() {
        return Err(Error::Model(format!(
            "DeepSeek-V4 {label} batched RMS length mismatch: tokens={tokens} input={} weight={}",
            input.len(),
            weight.len()
        )));
    }
    let mut out = Vec::with_capacity(input.len());
    for token in 0..tokens {
        let row = &input[token * weight.len()..(token + 1) * weight.len()];
        let normalized = operators.rms_norm(row, weight, eps, label)?;
        out.extend_from_slice(&normalized);
    }
    Ok(out)
}

fn window_topk_indices_prefill(window_size: usize, tokens: usize) -> Vec<isize> {
    let cols = tokens.min(window_size);
    let mut out = vec![-1; tokens * cols];
    for token in 0..tokens {
        let first = (token + 1).saturating_sub(window_size);
        for col in 0..cols {
            let idx = first + col;
            if idx <= token {
                out[token * cols + col] = idx as isize;
            }
        }
    }
    out
}

fn compress_topk_indices_prefill(
    ratio: usize,
    tokens: usize,
    offset: usize,
) -> (Vec<isize>, usize) {
    if ratio == 0 {
        return (Vec::new(), 0);
    }
    let cols = tokens / ratio;
    if cols == 0 {
        return (Vec::new(), 0);
    }
    let mut out = vec![-1; tokens * cols];
    for token in 0..tokens {
        let visible = (token + 1) / ratio;
        for idx in 0..cols {
            if idx < visible {
                out[token * cols + idx] = (offset + idx) as isize;
            }
        }
    }
    (out, cols)
}

fn concat_topk_rows(
    left: &[isize],
    left_cols: usize,
    right: &[isize],
    right_cols: usize,
    tokens: usize,
) -> Result<Vec<isize>> {
    if left.len() != tokens * left_cols || right.len() != tokens * right_cols {
        return Err(Error::Model(format!(
            "DeepSeek-V4 top-k concat shape mismatch: tokens={tokens} left={} left_cols={left_cols} right={} right_cols={right_cols}",
            left.len(),
            right.len()
        )));
    }
    let mut out = Vec::with_capacity(tokens * (left_cols + right_cols));
    for token in 0..tokens {
        out.extend_from_slice(&left[token * left_cols..(token + 1) * left_cols]);
        out.extend_from_slice(&right[token * right_cols..(token + 1) * right_cols]);
    }
    Ok(out)
}

fn indexer_topk_indices_prefill(
    indexer: &DeepSeekV4IndexerPayload,
    cfg: DeepSeekV4AttentionConfig,
    q_latents: &[f32],
    hidden: &[f32],
    indexer_compressed: &[f32],
    offset: usize,
    operators: &mut DeepSeekV4OperatorContext,
) -> Result<(Vec<isize>, usize)> {
    let tokens = hidden.len() / cfg.hidden_size;
    let compressed_len = indexer_compressed.len() / cfg.index_head_dim;
    let cols = cfg.index_topk.min(compressed_len);
    if cols == 0 {
        return Ok((Vec::new(), 0));
    }
    if hidden.len() != tokens * cfg.hidden_size
        || q_latents.len() != tokens * cfg.q_lora_rank
        || indexer_compressed.len() != compressed_len * cfg.index_head_dim
    {
        return Err(Error::Model(format!(
            "DeepSeek-V4 indexer prefill shape mismatch: tokens={tokens} hidden={} q_latents={} compressed={}",
            hidden.len(),
            q_latents.len(),
            indexer_compressed.len()
        )));
    }

    let mut out = vec![-1; tokens * cols];
    for token in 0..tokens {
        let q_latent = &q_latents[token * cfg.q_lora_rank..(token + 1) * cfg.q_lora_rank];
        let mut query = operators.linear_matvec(&indexer.wq_b, q_latent)?;
        apply_rotary_tail_scaled(
            &mut query,
            cfg.index_n_heads,
            cfg.index_head_dim,
            cfg.rope_head_dim.min(cfg.index_head_dim),
            token,
            cfg.rope_params(),
            false,
        )?;
        let hidden_row = &hidden[token * cfg.hidden_size..(token + 1) * cfg.hidden_size];
        let mut weights = operators.linear_matvec(&indexer.weights_proj, hidden_row)?;
        let scale = (cfg.index_head_dim as f32).powf(-0.5) * (cfg.index_n_heads as f32).powf(-0.5);
        for weight in &mut weights {
            *weight *= scale;
        }

        let visible = (token + 1) / cfg.compress_ratio;
        if visible == 0 {
            continue;
        }
        let mut scores = vec![f32::NEG_INFINITY; compressed_len];
        for idx in 0..compressed_len.min(visible) {
            let kv = &indexer_compressed[idx * cfg.index_head_dim..(idx + 1) * cfg.index_head_dim];
            let mut score = 0.0f32;
            for head in 0..cfg.index_n_heads {
                let q = &query[head * cfg.index_head_dim..(head + 1) * cfg.index_head_dim];
                score += dot(q, kv).max(0.0) * weights[head];
            }
            scores[idx] = score;
        }
        let mut order = (0..compressed_len.min(visible)).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        for (slot, idx) in order.into_iter().take(cols).enumerate() {
            if scores[idx].is_finite() {
                out[token * cols + slot] = (offset + idx) as isize;
            }
        }
    }
    Ok((out, cols))
}

fn compress_rows_softmax(
    kv_rows: &[f32],
    score_rows: &[f32],
    rows: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if rows == 0
        || head_dim == 0
        || kv_rows.len() != rows * head_dim
        || score_rows.len() != rows * head_dim
    {
        return Err(Error::Model(format!(
            "DeepSeek-V4 compressor row shape mismatch: rows={rows} head_dim={head_dim} kv={} score={}",
            kv_rows.len(),
            score_rows.len()
        )));
    }
    let mut out = vec![0.0f32; head_dim];
    for dim in 0..head_dim {
        let mut max_score = f32::NEG_INFINITY;
        for row in 0..rows {
            max_score = max_score.max(score_rows[row * head_dim + dim]);
        }
        if !max_score.is_finite() {
            continue;
        }
        let mut denom = 0.0f32;
        for row in 0..rows {
            let score = score_rows[row * head_dim + dim];
            if score.is_finite() {
                denom += (score - max_score).exp();
            }
        }
        if denom == 0.0 || !denom.is_finite() {
            return Err(Error::Model(
                "DeepSeek-V4 compressor softmax denominator is invalid".into(),
            ));
        }
        for row in 0..rows {
            let score = score_rows[row * head_dim + dim];
            if score.is_finite() {
                let weight = (score - max_score).exp() / denom;
                out[dim] += weight * kv_rows[row * head_dim + dim];
            }
        }
    }
    Ok(out)
}

fn indexer_topk_indices(
    indexer: &DeepSeekV4IndexerPayload,
    cfg: DeepSeekV4AttentionConfig,
    q_latent: &[f32],
    hidden: &[f32],
    position: usize,
    indexer_compressed: &[f32],
    offset: usize,
    operators: &mut DeepSeekV4OperatorContext,
) -> Result<Vec<isize>> {
    let compressed_len = indexer_compressed.len() / cfg.index_head_dim;
    if compressed_len == 0 {
        return Ok(Vec::new());
    }
    if indexer_compressed.len() != compressed_len * cfg.index_head_dim {
        return Err(Error::Model(
            "DeepSeek-V4 indexer compressed cache length is not divisible by index_head_dim".into(),
        ));
    }
    let mut query = operators.linear_matvec(&indexer.wq_b, q_latent)?;
    apply_rotary_tail_scaled(
        &mut query,
        cfg.index_n_heads,
        cfg.index_head_dim,
        cfg.rope_head_dim.min(cfg.index_head_dim),
        position,
        cfg.rope_params(),
        false,
    )?;
    let mut weights = operators.linear_matvec(&indexer.weights_proj, hidden)?;
    let scale = (cfg.index_head_dim as f32).powf(-0.5) * (cfg.index_n_heads as f32).powf(-0.5);
    for weight in &mut weights {
        *weight *= scale;
    }
    let mut scores = vec![0.0f32; compressed_len];
    for token in 0..compressed_len {
        let kv = &indexer_compressed[token * cfg.index_head_dim..(token + 1) * cfg.index_head_dim];
        let mut score = 0.0f32;
        for head in 0..cfg.index_n_heads {
            let q = &query[head * cfg.index_head_dim..(head + 1) * cfg.index_head_dim];
            score += dot(q, kv).max(0.0) * weights[head];
        }
        scores[token] = score;
    }
    let take = cfg.index_topk.min(compressed_len);
    let mut order = (0..compressed_len).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    Ok(order
        .into_iter()
        .take(take)
        .map(|idx| (offset + idx) as isize)
        .collect())
}

fn grouped_output_a(
    output_a: &SourceLinearPayload,
    context: &[f32],
    cfg: DeepSeekV4AttentionConfig,
    layer: usize,
) -> Result<Vec<f32>> {
    if context.len() != cfg.q_full_dim() {
        return Err(Error::Model(format!(
            "DeepSeek-V4 layer {layer} context length mismatch: expected {}, got {}",
            cfg.q_full_dim(),
            context.len()
        )));
    }
    check_linear(
        layer,
        "wo_a",
        output_a,
        cfg.output_latent_dim(),
        cfg.output_group_input_dim(),
    )?;
    let weights = output_a.reference_weights_f32()?;
    let group_in = cfg.output_group_input_dim();
    let mut out = vec![0.0; cfg.output_latent_dim()];
    for group in 0..cfg.o_groups {
        let context_start = group * group_in;
        let context_group = &context[context_start..context_start + group_in];
        for rank in 0..cfg.o_lora_rank {
            let row = group * cfg.o_lora_rank + rank;
            let weight_row = &weights[row * group_in..(row + 1) * group_in];
            out[row] = dot(weight_row, context_group);
        }
    }
    Ok(out)
}

fn rms_norm(input: &[f32], weight: &[f32], eps: f32, label: &str) -> Result<Vec<f32>> {
    if input.len() != weight.len() || input.is_empty() {
        return Err(Error::Model(format!(
            "DeepSeek-V4 {label} RMS length mismatch: input={}, weight={}",
            input.len(),
            weight.len()
        )));
    }
    let scale = (input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32 + eps)
        .sqrt()
        .recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * scale * weight)
        .collect())
}

fn rms_norm_heads_in_place(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    eps: f32,
    layer: usize,
) -> Result<()> {
    if values.len() != heads * head_dim {
        return Err(Error::Model(format!(
            "DeepSeek-V4 layer {layer} query length mismatch: expected {}, got {}",
            heads * head_dim,
            values.len()
        )));
    }
    for head in 0..heads {
        let row = &mut values[head * head_dim..(head + 1) * head_dim];
        let scale = (row.iter().map(|value| value * value).sum::<f32>() / head_dim as f32 + eps)
            .sqrt()
            .recip();
        for value in row {
            *value *= scale;
        }
    }
    Ok(())
}

fn apply_rotary_tail(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    theta: f32,
    inverse: bool,
) -> Result<()> {
    apply_rotary_tail_scaled(
        values,
        heads,
        head_dim,
        rope_dim,
        position,
        DeepSeekV4RopeParams::plain(theta),
        inverse,
    )
}

fn apply_rotary_tail_scaled(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    rope: DeepSeekV4RopeParams,
    inverse: bool,
) -> Result<()> {
    if rope_dim == 0 {
        return Ok(());
    }
    if rope_dim > head_dim || rope_dim % 2 != 0 || values.len() != heads * head_dim {
        return Err(Error::Model(format!(
            "DeepSeek-V4 rotary shape mismatch: values={}, heads={heads}, head_dim={head_dim}, rope_dim={rope_dim}",
            values.len()
        )));
    }
    let tail_start = head_dim - rope_dim;
    for head in 0..heads {
        let base = head * head_dim + tail_start;
        for pair in 0..rope_dim / 2 {
            let freq = yarn_frequency(pair, rope_dim, rope);
            let angle = position as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let sin = if inverse { -sin } else { sin };
            let x0 = values[base + 2 * pair];
            let x1 = values[base + 2 * pair + 1];
            values[base + 2 * pair] = x0 * cos - x1 * sin;
            values[base + 2 * pair + 1] = x0 * sin + x1 * cos;
        }
    }
    Ok(())
}

fn yarn_frequency(pair: usize, rope_dim: usize, rope: DeepSeekV4RopeParams) -> f32 {
    let base_freq = 1.0 / rope.theta.powf((2 * pair) as f32 / rope_dim as f32);
    if rope.original_seq_len == 0 || rope.factor == 1.0 {
        return base_freq;
    }
    let (low, high) = yarn_correction_range(
        rope.beta_fast as f32,
        rope.beta_slow as f32,
        rope_dim,
        rope.theta,
        rope.original_seq_len as f32,
    );
    let ramp = yarn_linear_ramp(pair as f32, low as f32, high as f32);
    let smooth = 1.0 - ramp;
    base_freq / rope.factor * (1.0 - smooth) + base_freq * smooth
}

fn yarn_correction_range(
    low_rot: f32,
    high_rot: f32,
    dim: usize,
    base: f32,
    max_position: f32,
) -> (usize, usize) {
    let low = yarn_correction_dim(low_rot, dim, base, max_position).floor() as isize;
    let high = yarn_correction_dim(high_rot, dim, base, max_position).ceil() as isize;
    (
        low.max(0) as usize,
        high.min(dim as isize - 1).max(0) as usize,
    )
}

fn yarn_correction_dim(num_rotations: f32, dim: usize, base: f32, max_position: f32) -> f32 {
    dim as f32 * (max_position / (num_rotations * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

fn yarn_linear_ramp(value: f32, min: f32, mut max: f32) -> f32 {
    if (min - max).abs() < f32::EPSILON {
        max += 0.001;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

fn unique_top_level_slice(
    model_dir: &Path,
    inventory: &HfSafetensorsInventory,
    role: TensorRole,
) -> Result<SourceTensorSlice> {
    let tensors = inventory
        .tensors
        .iter()
        .filter(|tensor| tensor.role == role)
        .collect::<Vec<_>>();
    match tensors.as_slice() {
        [tensor] => Ok(SourceTensorSlice::from_hf_inventory(model_dir, tensor)),
        [] => Err(Error::Model(format!(
            "DeepSeek-V4 missing top-level tensor role {role}"
        ))),
        _ => Err(Error::Model(format!(
            "DeepSeek-V4 expected exactly one top-level tensor role {role}, got {}",
            tensors.len()
        ))),
    }
}

fn read_named_vector_f32(
    model_dir: &Path,
    inventory: &HfSafetensorsInventory,
    reader: &SourceTensorReader,
    name: &str,
    role: TensorRole,
) -> Result<Vec<f32>> {
    let tensor = inventory_tensor(inventory, name)?;
    let mut slice = SourceTensorSlice::from_hf_inventory(model_dir, tensor);
    slice.role = role;
    decode_vector_f32(&reader.read_slice(&slice)?)
}

fn inventory_tensor<'a>(
    inventory: &'a HfSafetensorsInventory,
    name: &str,
) -> Result<&'a HfSafetensorsTensorInfo> {
    inventory
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::Model(format!("DeepSeek-V4 missing tensor '{name}'")))
}

fn decode_vector_f32(payload: &SourceTensorPayload) -> Result<Vec<f32>> {
    if payload.slice.shape.len() != 1 {
        return Err(Error::Model(format!(
            "DeepSeek-V4 source vector '{}' expects 1D shape, got {:?}",
            payload.slice.name, payload.slice.shape
        )));
    }
    decode_tensor_f32(payload)
}

fn decode_tensor_f32(payload: &SourceTensorPayload) -> Result<Vec<f32>> {
    let expected = payload.slice.element_count()?;
    match payload.slice.dtype {
        SourceDType::F32 => {
            if payload.bytes.len() != expected * 4 {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 F32 tensor '{}' byte length mismatch",
                    payload.slice.name
                )));
            }
            Ok(payload
                .bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        SourceDType::Bf16 => {
            if payload.bytes.len() != expected * 2 {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 BF16 tensor '{}' byte length mismatch",
                    payload.slice.name
                )));
            }
            Ok(payload
                .bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                    f32::from_bits(bits << 16)
                })
                .collect())
        }
        _ => Err(Error::Model(format!(
            "DeepSeek-V4 source tensor '{}' has unsupported vector dtype {}",
            payload.slice.name,
            payload.slice.dtype.as_str()
        ))),
    }
}

fn usize_key(json: &serde_json::Value, keys: &[&str]) -> Option<usize> {
    keys.iter().find_map(|key| {
        json.get(*key)
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
    })
}

fn f32_key(json: &serde_json::Value, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|key| {
        json.get(*key)
            .and_then(|value| value.as_f64())
            .map(|value| value as f32)
    })
}

fn rank_logits_desc(left: &DeepSeekV4Logit, right: &DeepSeekV4Logit) -> Ordering {
    right
        .logit
        .total_cmp(&left.logit)
        .then_with(|| left.token_id.cmp(&right.token_id))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(feature = "cuda")]
fn accumulate_output(target: &mut Option<Vec<f32>>, value: Vec<f32>) -> Result<()> {
    if let Some(target) = target {
        if target.len() != value.len() {
            return Err(Error::Model(format!(
                "output accumulation length mismatch: accumulated={}, next={}",
                target.len(),
                value.len()
            )));
        }
        for (dst, value) in target.iter_mut().zip(value) {
            *dst += value;
        }
    } else {
        *target = Some(value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::expert_executor::CpuReferenceExpertExecutor;
    use crate::expert_streaming::{
        ExpertId, ExpertMatrixKind, ExpertSource, ExpertTensorComponent, ExpertTensorKey,
        ExpertTensorPayload, ExpertTensorSlice,
    };
    use crate::source_linear::SourceLinearPayload;

    #[test]
    fn attention_shape_contract_accepts_official_dimensions() {
        let cfg = official_tiny_cfg();
        let payload = attention_payload_for_cfg(cfg);
        let attention = DeepSeekV4Attention::new(0, cfg, payload).unwrap();
        assert_eq!(attention.config.output_group_input_dim(), 4096);
        assert_eq!(attention.config.output_latent_dim(), 8192);
    }

    #[test]
    fn non_compressed_attention_decode_runs_grouped_output_projection() {
        let cfg = DeepSeekV4AttentionConfig {
            hidden_size: 4,
            num_heads: 2,
            head_dim: 2,
            q_lora_rank: 4,
            rope_head_dim: 2,
            o_groups: 1,
            o_lora_rank: 4,
            window_size: 4,
            compress_ratio: 0,
            norm_eps: 1e-6,
            rope_theta: 10000.0,
            compress_rope_theta: 160000.0,
            original_seq_len: 0,
            rope_factor: 1.0,
            beta_fast: 32,
            beta_slow: 1,
            index_n_heads: 2,
            index_head_dim: 2,
            index_topk: 4,
        };
        let payload = attention_payload_for_small_cfg(cfg);
        let attention = DeepSeekV4Attention::new(0, cfg, payload).unwrap();
        let mut cache = DeepSeekV4WindowKvCache::new(cfg.window_size, cfg.head_dim);
        let out = attention
            .decode_step_no_compress(&mut cache, &[1.0, 0.0, 0.0, 0.0], 0)
            .unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(cache.len(), 1);
        assert!(out.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn compressed_attention_decode_reference_updates_compressed_cache() {
        let cfg = DeepSeekV4AttentionConfig {
            hidden_size: 4,
            num_heads: 2,
            head_dim: 2,
            q_lora_rank: 4,
            rope_head_dim: 0,
            o_groups: 1,
            o_lora_rank: 4,
            window_size: 4,
            compress_ratio: 2,
            norm_eps: 1e-6,
            rope_theta: 10000.0,
            compress_rope_theta: 160000.0,
            original_seq_len: 0,
            rope_factor: 1.0,
            beta_fast: 32,
            beta_slow: 1,
            index_n_heads: 2,
            index_head_dim: 2,
            index_topk: 4,
        };
        let payload = attention_payload_for_small_cfg(cfg);
        let compressed = DeepSeekV4CompressedAttentionPayload {
            compressor: tiny_compressor_payload(cfg.compress_ratio, cfg.hidden_size, cfg.head_dim),
            indexer: None,
        };
        let attention =
            DeepSeekV4Attention::new_with_compressed(2, cfg, payload, Some(compressed)).unwrap();
        let mut cache = DeepSeekV4AttentionCache::new(cfg);
        let first = attention
            .decode_step_reference(&mut cache, &[1.0, 0.0, 0.0, 0.0], 0)
            .unwrap();
        assert_eq!(first.len(), 4);
        assert_eq!(cache.compressed_len(), 0);
        let second = attention
            .decode_step_reference(&mut cache, &[0.0, 1.0, 0.0, 0.0], 1)
            .unwrap();
        assert_eq!(second.len(), 4);
        assert_eq!(cache.compressed_len(), 1);
        assert!(second.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn window_kv_indices_follow_official_ring_order_after_wrap() {
        let mut cache = DeepSeekV4WindowKvCache::new(4, 1);
        for pos in 0..6 {
            cache.append(pos, &[pos as f32]).unwrap();
        }
        assert_eq!(cache.topk_indices(5, 4), vec![2, 3, 0, 1]);
    }

    #[test]
    fn official_prefill_topk_helpers_mask_future_positions() {
        assert_eq!(
            window_topk_indices_prefill(4, 5),
            vec![
                0, -1, -1, -1, // token 0
                0, 1, -1, -1, // token 1
                0, 1, 2, -1, // token 2
                0, 1, 2, 3, // token 3
                1, 2, 3, 4, // token 4
            ]
        );
        let (compressed, cols) = compress_topk_indices_prefill(4, 5, 5);
        assert_eq!(cols, 1);
        assert_eq!(compressed, vec![-1, -1, -1, 5, 5]);
    }

    #[test]
    fn dsv4_layer_decode_step_runs_hc_attention_moe_shared_hc() {
        let dir = unique_temp_dir("ferrule-dsv4-layer-decode");
        std::fs::create_dir_all(&dir).unwrap();

        let hc_config = HyperConnectionConfig {
            hc_mult: 2,
            hidden_size: 32,
            sinkhorn_iters: 3,
            eps: 1e-6,
            norm_eps: 1e-6,
        };
        let attention_cfg = DeepSeekV4AttentionConfig {
            hidden_size: 32,
            num_heads: 1,
            head_dim: 32,
            q_lora_rank: 4,
            rope_head_dim: 0,
            o_groups: 1,
            o_lora_rank: 4,
            window_size: 4,
            compress_ratio: 0,
            norm_eps: 1e-6,
            rope_theta: 10000.0,
            compress_rope_theta: 160000.0,
            original_seq_len: 0,
            rope_factor: 1.0,
            beta_fast: 32,
            beta_slow: 1,
            index_n_heads: 2,
            index_head_dim: 2,
            index_topk: 4,
        };
        let layer = DeepSeekV4Layer {
            layer: 0,
            hc_config,
            attn_norm: vec![1.0; 32],
            ffn_norm: vec![1.0; 32],
            attention: DeepSeekV4Attention::new(
                0,
                attention_cfg,
                attention_payload_for_vertical_cfg(attention_cfg),
            )
            .unwrap(),
            hc_attention: zero_hc_weights(hc_config),
            hc_feed_forward: zero_hc_weights(hc_config),
            router: RouterSourcePayload {
                layer: 0,
                weight: f32_linear(TensorRole::RouterLogits, "router", 1, 32),
                bias: None,
                hash_table: Some(vec![0]),
                hash_rows: 1,
                hash_cols: 1,
            },
            shared_ffn: tiny_shared_ffn_32(),
            router_policy: ExpertRouterPolicy::sqrt_softplus_hash(1, 1.0),
        };

        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first(1));
        register_tiny_expert(&dir, &mut planner, 0, 0, 0x42, 0x43, 0x22);
        let mut state = DeepSeekV4LayerState {
            kv: DeepSeekV4AttentionCache::new(attention_cfg),
            expert_planner: planner,
            expert_handles: CpuExpertHandleStore::new(),
        };
        let mut hc_state = vec![0.0f32; hc_config.hc_hidden_size()];
        hc_state[0] = 2.0;
        hc_state[33] = 3.0;

        let output = layer
            .decode_step_reference(
                &mut state,
                &hc_state,
                0,
                0,
                &[],
                &ExpertStreamingReader::new(4096),
                &CpuReferenceExpertExecutor::new(10.0),
            )
            .unwrap();
        assert_eq!(output.attention_hidden.len(), 32);
        assert_eq!(output.feed_forward_hidden.len(), 32);
        assert_eq!(output.hc_state.len(), hc_config.hc_hidden_size());
        assert_eq!(output.moe.routes.len(), 1);
        assert_eq!(output.moe.routes[0].expert, 0);
        assert_eq!(state.kv.len(), 1);
        assert!(output.hc_state.iter().all(|value| value.is_finite()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn official_tiny_cfg() -> DeepSeekV4AttentionConfig {
        DeepSeekV4AttentionConfig {
            hidden_size: deepseek_v4::HIDDEN_SIZE,
            num_heads: deepseek_v4::NUM_HEADS,
            head_dim: deepseek_v4::HEAD_DIM,
            q_lora_rank: deepseek_v4::Q_LORA_RANK,
            rope_head_dim: deepseek_v4::QK_ROPE_HEAD_DIM,
            o_groups: deepseek_v4::O_GROUPS,
            o_lora_rank: deepseek_v4::O_LORA_RANK,
            window_size: deepseek_v4::SLIDING_WINDOW,
            compress_ratio: 0,
            norm_eps: deepseek_v4::RMS_NORM_EPS,
            rope_theta: deepseek_v4::ROPE_THETA,
            compress_rope_theta: deepseek_v4::COMPRESS_ROPE_THETA,
            original_seq_len: deepseek_v4::ORIGINAL_MAX_POSITION_EMBEDDINGS,
            rope_factor: deepseek_v4::ROPE_FACTOR,
            beta_fast: deepseek_v4::ROPE_BETA_FAST,
            beta_slow: deepseek_v4::ROPE_BETA_SLOW,
            index_n_heads: deepseek_v4::INDEX_N_HEADS,
            index_head_dim: deepseek_v4::INDEX_HEAD_DIM,
            index_topk: deepseek_v4::INDEX_TOPK,
        }
    }

    fn attention_payload_for_cfg(cfg: DeepSeekV4AttentionConfig) -> AttentionSourcePayload {
        AttentionSourcePayload {
            layer: 0,
            query_a: f32_linear(
                TensorRole::AttentionLatentQueryA,
                "wq_a",
                cfg.q_lora_rank,
                cfg.hidden_size,
            ),
            query_b: f32_linear(
                TensorRole::AttentionLatentQueryB,
                "wq_b",
                cfg.q_full_dim(),
                cfg.q_lora_rank,
            ),
            key_value: f32_linear(
                TensorRole::AttentionLatentKv,
                "wkv",
                cfg.head_dim,
                cfg.hidden_size,
            ),
            output_a: f32_linear(
                TensorRole::AttentionLatentOutputA,
                "wo_a",
                cfg.output_latent_dim(),
                cfg.output_group_input_dim(),
            ),
            output_b: f32_linear(
                TensorRole::AttentionLatentOutputB,
                "wo_b",
                cfg.hidden_size,
                cfg.output_latent_dim(),
            ),
            query_norm: vec![1.0; cfg.q_lora_rank],
            key_value_norm: vec![1.0; cfg.head_dim],
            attention_sink: vec![0.0; cfg.num_heads],
            auxiliary: Vec::new(),
        }
    }

    fn attention_payload_for_small_cfg(cfg: DeepSeekV4AttentionConfig) -> AttentionSourcePayload {
        AttentionSourcePayload {
            layer: 0,
            query_a: identity_linear(TensorRole::AttentionLatentQueryA, "wq_a", 4),
            query_b: identity_linear(TensorRole::AttentionLatentQueryB, "wq_b", 4),
            key_value: f32_linear_values(
                TensorRole::AttentionLatentKv,
                "wkv",
                2,
                4,
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
            output_a: identity_linear(TensorRole::AttentionLatentOutputA, "wo_a", 4),
            output_b: identity_linear(TensorRole::AttentionLatentOutputB, "wo_b", 4),
            query_norm: vec![1.0; cfg.q_lora_rank],
            key_value_norm: vec![1.0; cfg.head_dim],
            attention_sink: vec![0.0; cfg.num_heads],
            auxiliary: Vec::new(),
        }
    }

    fn attention_payload_for_vertical_cfg(
        cfg: DeepSeekV4AttentionConfig,
    ) -> AttentionSourcePayload {
        AttentionSourcePayload {
            layer: 0,
            query_a: f32_linear_values(
                TensorRole::AttentionLatentQueryA,
                "wq_a",
                cfg.q_lora_rank,
                cfg.hidden_size,
                &one_hot_rows(
                    cfg.q_lora_rank,
                    cfg.hidden_size,
                    &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
                ),
            ),
            query_b: f32_linear_values(
                TensorRole::AttentionLatentQueryB,
                "wq_b",
                cfg.q_full_dim(),
                cfg.q_lora_rank,
                &one_hot_rows(
                    cfg.q_full_dim(),
                    cfg.q_lora_rank,
                    &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
                ),
            ),
            key_value: identity_linear(TensorRole::AttentionLatentKv, "wkv", cfg.head_dim),
            output_a: f32_linear_values(
                TensorRole::AttentionLatentOutputA,
                "wo_a",
                cfg.output_latent_dim(),
                cfg.output_group_input_dim(),
                &one_hot_rows(
                    cfg.output_latent_dim(),
                    cfg.output_group_input_dim(),
                    &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
                ),
            ),
            output_b: f32_linear_values(
                TensorRole::AttentionLatentOutputB,
                "wo_b",
                cfg.hidden_size,
                cfg.output_latent_dim(),
                &one_hot_rows(
                    cfg.hidden_size,
                    cfg.output_latent_dim(),
                    &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
                ),
            ),
            query_norm: vec![1.0; cfg.q_lora_rank],
            key_value_norm: vec![1.0; cfg.head_dim],
            attention_sink: vec![0.0; cfg.num_heads],
            auxiliary: Vec::new(),
        }
    }

    fn identity_linear(role: TensorRole, name: &str, dim: usize) -> SourceLinearPayload {
        let mut values = vec![0.0; dim * dim];
        for i in 0..dim {
            values[i * dim + i] = 1.0;
        }
        f32_linear_values(role, name, dim, dim, &values)
    }

    fn f32_linear(role: TensorRole, name: &str, out: usize, input: usize) -> SourceLinearPayload {
        f32_linear_values(role, name, out, input, &vec![0.0; out * input])
    }

    fn f32_linear_values(
        role: TensorRole,
        name: &str,
        out: usize,
        input: usize,
        values: &[f32],
    ) -> SourceLinearPayload {
        assert_eq!(values.len(), out * input);
        SourceLinearPayload::from_weight_and_scale(
            role,
            SourceTensorPayload {
                slice: SourceTensorSlice {
                    name: format!("{name}.weight"),
                    role: TensorRole::Unknown,
                    path: PathBuf::from("synthetic.safetensors"),
                    offset: 0,
                    bytes: (values.len() * 4) as u64,
                    dtype: SourceDType::F32,
                    shape: vec![out, input],
                },
                bytes: values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            },
            None,
        )
        .unwrap()
    }

    fn tiny_compressor_payload(
        ratio: usize,
        hidden_size: usize,
        head_dim: usize,
    ) -> DeepSeekV4CompressorPayload {
        DeepSeekV4CompressorPayload {
            compress_ratio: ratio,
            head_dim,
            overlap: false,
            rotate_for_indexer: false,
            ape: vec![0.0; ratio * head_dim],
            ape_rows: ratio,
            ape_cols: head_dim,
            norm: vec![1.0; head_dim],
            wkv: f32_linear_values(
                TensorRole::AttentionCompressor,
                "compressor.wkv",
                head_dim,
                hidden_size,
                &one_hot_rows(head_dim, hidden_size, &[(0, 0, 1.0), (1, 1, 1.0)]),
            ),
            wgate: f32_linear_values(
                TensorRole::AttentionCompressor,
                "compressor.wgate",
                head_dim,
                hidden_size,
                &vec![0.0; head_dim * hidden_size],
            ),
        }
    }

    fn tiny_shared_ffn_32() -> SwiGluFfnPayload {
        SwiGluFfnPayload {
            gate: f32_linear_values(
                TensorRole::SharedExpertGate,
                "shared_gate",
                1,
                32,
                &one_hot_rows(1, 32, &[(0, 0, 1.0)]),
            ),
            up: f32_linear_values(
                TensorRole::SharedExpertUp,
                "shared_up",
                1,
                32,
                &one_hot_rows(1, 32, &[(0, 1, 1.0)]),
            ),
            down: f32_linear_values(
                TensorRole::SharedExpertDown,
                "shared_down",
                32,
                1,
                &one_hot_rows(32, 1, &[(0, 0, 1.0)]),
            ),
            swiglu_limit: 10.0,
        }
    }

    fn zero_hc_weights(config: HyperConnectionConfig) -> HyperConnectionWeights {
        HyperConnectionWeights {
            function: vec![0.0; config.mix_hc() * config.hc_hidden_size()],
            scale: vec![1.0, 1.0, 1.0],
            base: vec![0.0; config.mix_hc()],
        }
    }

    fn one_hot_rows(rows: usize, cols: usize, entries: &[(usize, usize, f32)]) -> Vec<f32> {
        let mut values = vec![0.0f32; rows * cols];
        for &(row, col, value) in entries {
            values[row * cols + col] = value;
        }
        values
    }

    fn register_tiny_expert(
        dir: &Path,
        planner: &mut ExpertStreamingPlanner,
        layer: usize,
        expert: usize,
        gate_byte: u8,
        up_byte: u8,
        down_byte: u8,
    ) {
        let expert_id = ExpertId::new(layer, expert);
        let path = dir.join(format!("l{layer}e{expert}.bin"));
        let tensors = tiny_expert_tensors(expert_id, &path, gate_byte, up_byte, down_byte);
        let mut bytes = Vec::new();
        for tensor in &tensors {
            bytes.extend(&tensor.bytes);
        }
        std::fs::write(&path, bytes).unwrap();
        let mut offset = 0u64;
        let slices = tensors
            .into_iter()
            .map(|tensor| {
                let bytes = tensor.bytes.len() as u64;
                let slice = ExpertTensorSlice {
                    offset,
                    bytes,
                    ..tensor.slice
                };
                offset += bytes;
                slice
            })
            .collect();
        planner.register_source(expert_id, ExpertSource::LocalTensorSet { tensors: slices });
    }

    fn tiny_expert_tensors(
        expert: ExpertId,
        path: &Path,
        gate_byte: u8,
        up_byte: u8,
        down_byte: u8,
    ) -> Vec<ExpertTensorPayload> {
        vec![
            tiny_fp4_payload(expert, path, ExpertMatrixKind::Gate, gate_byte),
            tiny_scale_payload(expert, path, ExpertMatrixKind::Gate),
            tiny_fp4_payload(expert, path, ExpertMatrixKind::Up, up_byte),
            tiny_scale_payload(expert, path, ExpertMatrixKind::Up),
            tiny_fp4_payload(expert, path, ExpertMatrixKind::Down, down_byte),
            tiny_scale_payload(expert, path, ExpertMatrixKind::Down),
        ]
    }

    fn tiny_fp4_payload(
        expert: ExpertId,
        path: &Path,
        matrix: ExpertMatrixKind,
        first_byte: u8,
    ) -> ExpertTensorPayload {
        let mut bytes = vec![0u8; 32 * 16];
        bytes[0] = first_byte;
        ExpertTensorPayload {
            slice: ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component: ExpertTensorComponent::Weight,
                path: path.to_path_buf(),
                offset: 0,
                bytes: bytes.len() as u64,
                dtype: "I8".into(),
                shape: vec![32, 16],
            },
            bytes,
        }
    }

    fn tiny_scale_payload(
        expert: ExpertId,
        path: &Path,
        matrix: ExpertMatrixKind,
    ) -> ExpertTensorPayload {
        let bytes = vec![127u8; 32];
        ExpertTensorPayload {
            slice: ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component: ExpertTensorComponent::Scale,
                path: path.to_path_buf(),
                offset: 0,
                bytes: bytes.len() as u64,
                dtype: "F8_E8M0".into(),
                shape: vec![32, 1],
            },
            bytes,
        }
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nonce}"))
    }
}
