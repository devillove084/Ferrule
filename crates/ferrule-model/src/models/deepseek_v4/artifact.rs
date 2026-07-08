//! DeepSeek-V4 artifact model: HF weight loading and tensor binding.

use std::path::Path;

use crate::artifact::binding::{
    bind_attention_from_hf, bind_hyper_connection_from_hf, bind_hyper_connection_head_from_hf,
    bind_router_from_hf, bind_shared_swiglu_ffn_from_hf,
};
#[cfg(any(feature = "cuda", test))]
use crate::artifact::linear::ArtifactLinearFormat;
use crate::artifact::linear::ArtifactLinearPayload;
#[cfg(feature = "cuda")]
use crate::artifact::tensor::ArtifactDType;
use crate::artifact::tensor::{ArtifactTensorReader, ArtifactTensorSlice};
use crate::hyper_connection::HyperConnectionHeadWeights;
use crate::moe::handle::CpuExpertHandleStore;
use crate::moe::routing::ExpertRouterPolicy;
use crate::moe::streaming::{ExpertStreamingPlanner, ExpertStreamingPolicy};
use crate::runner::ModelInfo;
use crate::semantic::HyperConnectionStage;
use crate::tokenizer::TokenizerHandle;
use crate::{HfSafetensorsInventory, ModelDescriptor, ModelFamily, TensorRole, WeightSource};
use ferrule_common::{Error, Result};

use super::attention::{
    DeepSeekV4Attention, DeepSeekV4AttentionCache, DeepSeekV4CompressedAttentionPayload,
};
use super::config::{
    with_deepseek_v4_attention_execution_policies, with_deepseek_v4_swiglu_execution_policies,
    DeepSeekV4Config,
};
use super::helpers::{
    decode_tensor_f32, decode_vector_f32, rank_logits_desc, read_named_vector_f32,
    unique_top_level_slice,
};
use super::layer::{DeepSeekV4Layer, DeepSeekV4LayerState};
use super::operators::{DeepSeekV4Logit, DeepSeekV4OperatorBackend, DeepSeekV4OperatorContext};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactTensor2D {
    pub slice: ArtifactTensorSlice,
    pub rows: usize,
    pub cols: usize,
}

impl ArtifactTensor2D {
    pub fn from_slice(slice: ArtifactTensorSlice, label: &str) -> Result<Self> {
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

pub struct DeepSeekV4ArtifactModel {
    pub descriptor: ModelDescriptor,
    pub config: DeepSeekV4Config,
    pub tokenizer: TokenizerHandle,
    pub embedding: ArtifactTensor2D,
    pub output_norm: Vec<f32>,
    pub output_head: ArtifactTensor2D,
    pub hc_head: HyperConnectionHeadWeights,
    inventory: HfSafetensorsInventory,
    max_tensor_bytes: u64,
}

impl DeepSeekV4ArtifactModel {
    pub fn load_hf(model_dir: &Path) -> Result<Self> {
        Self::load_hf_with_limit(model_dir, 128 * 1024 * 1024)
    }

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
        let reader = ArtifactTensorReader::new(max_tensor_bytes);
        let tokenizer = TokenizerHandle::load(model_dir)?;
        let embedding = ArtifactTensor2D::from_slice(
            unique_top_level_slice(model_dir, &inventory, TensorRole::TokenEmbedding)?,
            "token embedding",
        )?;
        let output_norm = decode_vector_f32(&reader.read_slice(&unique_top_level_slice(
            model_dir,
            &inventory,
            TensorRole::OutputNorm,
        )?)?)?;
        let output_head = ArtifactTensor2D::from_slice(
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
        let mut operators = DeepSeekV4OperatorContext::new(DeepSeekV4OperatorBackend::Cpu)?;
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
            if operators.backend() == DeepSeekV4OperatorBackend::Cuda {
                let state_buf = operators.cuda_upload_f32(hc_state)?;
                let hidden_buf = operators.cuda_hc_head_from_device(
                    &state_buf,
                    tokens,
                    self.config.hc_config(),
                    &self.hc_head,
                )?;
                operators.cuda_download_f32(&hidden_buf)?
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
        let mut operators = DeepSeekV4OperatorContext::new(DeepSeekV4OperatorBackend::Cpu)?;
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

    pub fn logits_for_hidden_chunked(&self, hidden: &[f32], chunk_rows: usize) -> Result<Vec<f32>> {
        let mut operators = DeepSeekV4OperatorContext::new(DeepSeekV4OperatorBackend::Cpu)?;
        self.logits_for_hidden_chunked_with_operators(hidden, chunk_rows, &mut operators)
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

    pub fn topk_logits_for_hidden(
        &self,
        hidden: &[f32],
        top_k: usize,
        chunk_rows: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        let mut operators = DeepSeekV4OperatorContext::new(DeepSeekV4OperatorBackend::Cpu)?;
        self.topk_logits_for_hidden_with_operators(hidden, top_k, chunk_rows, &mut operators)
    }

    pub(crate) fn topk_logits_for_hidden_with_operators(
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
        let reader = ArtifactTensorReader::new(self.max_tensor_bytes);
        if operators.backend() == DeepSeekV4OperatorBackend::Cuda {
            return operators.output_head_topk_chunks(
                &self.output_head.slice,
                hidden,
                top_k,
                chunk_rows,
                &reader,
            );
        }

        let mut top = Vec::<DeepSeekV4Logit>::new();
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
    ) -> Result<Vec<DeepSeekV4Logit>> {
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
        operators.cuda_output_head_topk_chunks_with_device(
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

    pub fn new_layer_state(
        &self,
        layer: usize,
        policy: ExpertStreamingPolicy,
    ) -> Result<DeepSeekV4LayerState> {
        let attention_config = self.config.attention_config_for_layer(layer)?;
        let routed = self
            .inventory
            .routed_expert_tensors()
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
        self.new_quality_first_layer_state_with_prefetch(layer, 0)
    }

    pub fn new_quality_first_layer_state_with_prefetch(
        &self,
        layer: usize,
        moe_prefetch_experts: usize,
    ) -> Result<DeepSeekV4LayerState> {
        self.new_quality_first_layer_state_with_residency(layer, moe_prefetch_experts, 0)
    }

    pub fn new_quality_first_layer_state_with_residency(
        &self,
        layer: usize,
        moe_prefetch_experts: usize,
        moe_hotset_experts: usize,
    ) -> Result<DeepSeekV4LayerState> {
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
            // Keep this default aligned with CUDA expert uploads: FP4 experts use
            // managed memory unless FERRULE_MANAGED_EXPERTS=0. With managed
            // memory, resident handles are cheap enough to keep until an explicit
            // hotset budget is requested, avoiding needless re-reads/re-uploads.
            let managed = std::env::var("FERRULE_MANAGED_EXPERTS")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(true);
            if managed {
                ExpertStreamingPolicy {
                    gpu_slots_per_layer: self.config.num_routed_experts,
                    prefetch_per_layer: 0,
                    preserve_artifact_quantization: true,
                    allow_cpu_staging: true,
                    allow_remote_sources: false,
                }
            } else {
                ExpertStreamingPolicy::quality_first_no_prefetch(self.config.num_experts_per_tok)
            }
        } else {
            ExpertStreamingPolicy::quality_first_with_prefetch(
                self.config.num_experts_per_tok,
                moe_prefetch_experts,
            )
        };
        self.new_layer_state(layer, policy)
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn artifact_linear_cache_key(linear: &ArtifactLinearPayload) -> String {
    artifact_linear_cache_key_from_parts(
        &linear.weight.slice,
        linear.scale.as_ref().map(|scale| &scale.slice),
        &linear.format,
    )
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn artifact_linear_cache_key_from_parts(
    weight: &ArtifactTensorSlice,
    scale: Option<&ArtifactTensorSlice>,
    format: &ArtifactLinearFormat,
) -> String {
    let scale_key = scale
        .map(artifact_tensor_slice_cache_key)
        .unwrap_or_else(|| "<none>".into());
    format!(
        "weight={}::scale={}::format={:?}",
        artifact_tensor_slice_cache_key(weight),
        scale_key,
        format
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn artifact_linear_row_cache_key(
    slice: &ArtifactTensorSlice,
    start_row: usize,
    row_count: usize,
) -> Result<String> {
    let row_slice = artifact_2d_row_slice_descriptor(slice, start_row, row_count)?;
    let format = artifact_unscaled_2d_linear_format(&row_slice)?;
    Ok(artifact_linear_cache_key_from_parts(
        &row_slice, None, &format,
    ))
}

#[cfg(feature = "cuda")]
fn artifact_2d_row_slice_descriptor(
    slice: &ArtifactTensorSlice,
    start_row: usize,
    row_count: usize,
) -> Result<ArtifactTensorSlice> {
    if slice.shape.len() != 2 {
        return Err(Error::Model(format!(
            "artifact tensor '{}' row descriptor expects 2D shape, got {:?}",
            slice.name, slice.shape
        )));
    }
    let rows = slice.shape[0];
    let cols = slice.shape[1];
    let end_row = start_row.checked_add(row_count).ok_or_else(|| {
        Error::Model(format!(
            "artifact tensor '{}' row descriptor overflows: start={start_row} count={row_count}",
            slice.name
        ))
    })?;
    if row_count == 0 || start_row >= rows || end_row > rows {
        return Err(Error::Model(format!(
            "artifact tensor '{}' invalid row descriptor: start={start_row} count={row_count} rows={rows}",
            slice.name
        )));
    }
    let elem_bytes = slice.dtype.element_size_bytes().ok_or_else(|| {
        Error::Model(format!(
            "artifact tensor '{}' has unknown dtype {} for row descriptor",
            slice.name,
            slice.dtype.as_str()
        ))
    })?;
    let row_bytes = cols.checked_mul(elem_bytes).ok_or_else(|| {
        Error::Model(format!(
            "artifact tensor '{}' row descriptor byte size overflows for cols={cols} elem_bytes={elem_bytes}",
            slice.name
        ))
    })?;
    let byte_offset = start_row.checked_mul(row_bytes).ok_or_else(|| {
        Error::Model(format!(
            "artifact tensor '{}' row descriptor offset overflows",
            slice.name
        ))
    })?;
    let bytes = row_count.checked_mul(row_bytes).ok_or_else(|| {
        Error::Model(format!(
            "artifact tensor '{}' row descriptor byte count overflows",
            slice.name
        ))
    })?;
    let mut row_slice = slice.clone();
    row_slice.offset = slice
        .offset
        .checked_add(byte_offset as u64)
        .ok_or_else(|| {
            Error::Model(format!(
                "artifact tensor '{}' absolute row descriptor offset overflows",
                slice.name
            ))
        })?;
    row_slice.bytes = bytes as u64;
    row_slice.shape = vec![row_count, cols];
    Ok(row_slice)
}

#[cfg(feature = "cuda")]
fn artifact_unscaled_2d_linear_format(slice: &ArtifactTensorSlice) -> Result<ArtifactLinearFormat> {
    if slice.shape.len() != 2 {
        return Err(Error::Model(format!(
            "artifact linear '{}' cache-key format expects 2D shape, got {:?}",
            slice.name, slice.shape
        )));
    }
    let out_features = slice.shape[0];
    let in_features = slice.shape[1];
    match slice.dtype {
        ArtifactDType::F32 => Ok(ArtifactLinearFormat::F32 {
            out_features,
            in_features,
        }),
        ArtifactDType::Bf16 => Ok(ArtifactLinearFormat::Bf16 {
            out_features,
            in_features,
        }),
        _ => Err(Error::Model(format!(
            "artifact linear '{}' dtype {} cannot be used as an unscaled output-head row chunk",
            slice.name,
            slice.dtype.as_str()
        ))),
    }
}

#[cfg(any(feature = "cuda", test))]
fn artifact_tensor_slice_cache_key(slice: &ArtifactTensorSlice) -> String {
    format!(
        "{}@{}+{}:{:?}:{:?}",
        slice.path.display(),
        slice.offset,
        slice.bytes,
        slice.dtype,
        slice.shape
    )
}
