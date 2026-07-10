//! DeepSeek-V4 attention: MLA, compressor, indexer, window KV cache.

use std::time::Instant;

use crate::artifact::binding::MlaAttentionArtifactPayload;
use crate::artifact::linear::ArtifactLinearPayload;
use crate::artifact::tensor::{ArtifactTensorReader, ArtifactTensorSlice};
use ferrule_common::{Error, Result};

use super::config::{DeepSeekV4AttentionConfig, DeepSeekV4RopeParams};
use super::helpers::{
    apply_rotary_tail, apply_rotary_tail_scaled, bind_aux_linear, check_len, check_linear,
    compress_rows_softmax, compress_topk_indices_prefill, concat_topk_rows, decode_tensor_f32,
    decode_vector_f32, indexer_topk_indices, indexer_topk_indices_prefill,
    quantize_attention_kv_for_qat_in_place, quantize_compressed_kv_for_qat_in_place,
    read_aux_tensor, read_aux_tensor_f32, two_dim_shape_from_payload, window_topk_indices_prefill,
};
use super::operators::{
    DeepSeekV4AttentionProfileStage, DeepSeekV4OperatorBackend, DeepSeekV4OperatorContext,
};
use crate::TensorRole;

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
    pub wkv: ArtifactLinearPayload,
    pub wgate: ArtifactLinearPayload,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4IndexerPayload {
    pub compressor: DeepSeekV4CompressorPayload,
    pub wq_b: ArtifactLinearPayload,
    pub weights_proj: ArtifactLinearPayload,
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
        auxiliary: &[ArtifactTensorSlice],
        reader: &ArtifactTensorReader,
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
        auxiliary: &[ArtifactTensorSlice],
        reader: &ArtifactTensorReader,
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
        auxiliary: &[ArtifactTensorSlice],
        reader: &ArtifactTensorReader,
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
    pub payload: MlaAttentionArtifactPayload,
    pub compressed: Option<DeepSeekV4CompressedAttentionPayload>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4AttentionGraphArena {
    hidden_q: ferrule_cuda::context::CudaF32Buffer,
    index_weight_input: ferrule_cuda::context::CudaF32Buffer,
    q_latent: ferrule_cuda::context::CudaF32Buffer,
    q_normed: ferrule_cuda::context::CudaF32Buffer,
    q_normed_for_indexer: ferrule_cuda::context::CudaF32Buffer,
    query: ferrule_cuda::context::CudaF32Buffer,
    query_normed: ferrule_cuda::context::CudaF32Buffer,
    kv: ferrule_cuda::context::CudaF32Buffer,
    kv_normed: ferrule_cuda::context::CudaF32Buffer,
    index_query: ferrule_cuda::context::CudaF32Buffer,
    index_weights: ferrule_cuda::context::CudaF32Buffer,
    topk: ferrule_cuda::context::CudaI32Buffer,
    empty_query: ferrule_cuda::context::CudaF32Buffer,
    empty_weights: ferrule_cuda::context::CudaF32Buffer,
    empty_kv: ferrule_cuda::context::CudaF32Buffer,
    context: ferrule_cuda::context::CudaF32Buffer,
    latent: ferrule_cuda::context::CudaF32Buffer,
    output: ferrule_cuda::context::CudaF32Buffer,
    hidden_size: usize,
    q_lora_rank: usize,
    q_full_dim: usize,
    output_latent_dim: usize,
    max_topk: usize,
    index_query_dim: usize,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4AttentionGraphArena {
    pub(crate) fn new(
        cfg: DeepSeekV4AttentionConfig,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Self> {
        let q_full_dim = cfg.q_full_dim();
        let output_latent_dim = cfg.output_latent_dim();
        let max_topk = cfg
            .window_size
            .checked_add(cfg.index_topk)
            .ok_or_else(|| Error::Model("DeepSeek-V4 graph attention topk overflow".into()))?;
        let index_query_dim = cfg
            .index_n_heads
            .checked_mul(cfg.index_head_dim)
            .ok_or_else(|| {
                Error::Model("DeepSeek-V4 graph attention index query overflow".into())
            })?;
        Ok(Self {
            hidden_q: operators.cuda_zero_f32(cfg.hidden_size)?,
            index_weight_input: operators.cuda_zero_f32(cfg.hidden_size)?,
            q_latent: operators.cuda_zero_f32(cfg.q_lora_rank)?,
            q_normed: operators.cuda_zero_f32(cfg.q_lora_rank)?,
            q_normed_for_indexer: operators.cuda_zero_f32(cfg.q_lora_rank)?,
            query: operators.cuda_zero_f32(q_full_dim)?,
            query_normed: operators.cuda_zero_f32(q_full_dim)?,
            kv: operators.cuda_zero_f32(cfg.head_dim)?,
            kv_normed: operators.cuda_zero_f32(cfg.head_dim)?,
            index_query: operators.cuda_zero_f32(index_query_dim)?,
            index_weights: operators.cuda_zero_f32(cfg.index_n_heads)?,
            topk: operators.cuda_zero_i32(max_topk)?,
            empty_query: operators.cuda_zero_f32(1)?,
            empty_weights: operators.cuda_zero_f32(1)?,
            empty_kv: operators.cuda_zero_f32(1)?,
            context: operators.cuda_zero_f32(q_full_dim)?,
            latent: operators.cuda_zero_f32(output_latent_dim)?,
            output: operators.cuda_zero_f32(cfg.hidden_size)?,
            hidden_size: cfg.hidden_size,
            q_lora_rank: cfg.q_lora_rank,
            q_full_dim,
            output_latent_dim,
            max_topk,
            index_query_dim,
        })
    }

    pub(crate) fn is_compatible(&self, cfg: DeepSeekV4AttentionConfig) -> bool {
        self.hidden_size == cfg.hidden_size
            && self.q_lora_rank == cfg.q_lora_rank
            && self.q_full_dim == cfg.q_full_dim()
            && self.output_latent_dim == cfg.output_latent_dim()
            && self.max_topk >= cfg.window_size.saturating_add(cfg.index_topk)
            && self.index_query_dim == cfg.index_n_heads.saturating_mul(cfg.index_head_dim)
    }

    pub(crate) fn output(&self) -> &ferrule_cuda::context::CudaF32Buffer {
        &self.output
    }
}

impl DeepSeekV4Attention {
    pub fn new(
        layer: usize,
        config: DeepSeekV4AttentionConfig,
        payload: MlaAttentionArtifactPayload,
    ) -> Result<Self> {
        Self::new_with_compressed(layer, config, payload, None)
    }

    pub fn new_with_compressed(
        layer: usize,
        config: DeepSeekV4AttentionConfig,
        payload: MlaAttentionArtifactPayload,
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
            return self.prefill_segment_with_operators(cache, hidden, start_pos, operators);
        }
        if hidden.is_empty() || !hidden.len().is_multiple_of(cfg.hidden_size) {
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

    pub fn prefill_segment_with_operators(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden: &[f32],
        start_pos: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        let cfg = self.config;
        if hidden.is_empty() || !hidden.len().is_multiple_of(cfg.hidden_size) {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} segment prefill hidden length mismatch: hidden={} dim={}",
                self.layer,
                hidden.len(),
                cfg.hidden_size
            )));
        }
        let tokens = hidden.len() / cfg.hidden_size;
        let mut out = Vec::with_capacity(tokens * cfg.hidden_size);
        for token in 0..tokens {
            let position = start_pos + token;
            let row = &hidden[token * cfg.hidden_size..(token + 1) * cfg.hidden_size];
            let attention = self.decode_step_with_operators(cache, row, position, operators)?;
            out.extend_from_slice(&attention);
        }
        Ok(out)
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
        operators.record_attention_call(self.layer, tokens);
        if cache.window.head_dim != cfg.head_dim || cache.window.window_size != cfg.window_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} KV cache shape mismatch: cache window={} head_dim={}, expected window={} head_dim={}",
                self.layer, cache.window.window_size, cache.window.head_dim, cfg.window_size, cfg.head_dim
            )));
        }

        let stage_start = Instant::now();
        let q_latents = operators.linear_rows(&self.payload.query_a, hidden, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qa,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let q_norm_name = format!("q_norm_L{}", self.layer);
        let q_latents = operators.rms_norm_rows(
            &q_latents,
            tokens,
            &self.payload.query_norm,
            cfg.norm_eps,
            &q_norm_name,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let mut queries = operators.linear_rows(&self.payload.query_b, &q_latents, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qb,
            stage_start,
        )?;
        let stage_start = Instant::now();
        operators.rms_norm_heads_in_place(
            &mut queries,
            tokens * cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
            self.layer,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QHeadNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        for token in 0..tokens {
            let position = start_pos + token;
            let row_start = token * cfg.q_full_dim();
            apply_rotary_tail_scaled(
                &mut queries[row_start..row_start + cfg.q_full_dim()],
                cfg.num_heads,
                cfg.head_dim,
                cfg.rope_head_dim,
                position,
                cfg.rope_params(),
                false,
            )?;
        }
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QRope,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let kv_rows = operators.linear_rows(&self.payload.key_value, hidden, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvProj,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let kv_norm_name = format!("kv_norm_L{}", self.layer);
        let mut values = operators.rms_norm_rows(
            &kv_rows,
            tokens,
            &self.payload.key_value_norm,
            cfg.norm_eps,
            &kv_norm_name,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        for token in 0..tokens {
            let position = start_pos + token;
            let row_start = token * cfg.head_dim;
            let kv = &mut values[row_start..row_start + cfg.head_dim];
            apply_rotary_tail_scaled(
                kv,
                1,
                cfg.head_dim,
                cfg.rope_head_dim,
                position,
                cfg.rope_params(),
                false,
            )?;
            quantize_attention_kv_for_qat_in_place(kv, cfg.head_dim, cfg.rope_head_dim)?;
        }
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvRopeQuant,
            stage_start,
        )?;
        let stage_start = Instant::now();
        for token in 0..tokens {
            let position = start_pos + token;
            let row_start = token * cfg.head_dim;
            cache
                .window
                .append(position, &values[row_start..row_start + cfg.head_dim])?;
        }
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvCacheAppend,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let topk_cols = tokens.min(cfg.window_size);
        let topk = window_topk_indices_prefill(cfg.window_size, tokens);
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::TopkBuild,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let context = operators.sparse_attention(
            &queries,
            &values,
            &topk,
            &self.payload.attention_sink,
            tokens,
            tokens,
            cfg.sparse_spec_with_topk(topk_cols),
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::SparseAttention,
            stage_start,
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
        operators.record_attention_call(self.layer, tokens);
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

        let stage_start = Instant::now();
        let q_latents = operators.linear_rows(&self.payload.query_a, hidden, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qa,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let q_norm_name = format!("q_norm_L{}", self.layer);
        let q_latents = operators.rms_norm_rows(
            &q_latents,
            tokens,
            &self.payload.query_norm,
            cfg.norm_eps,
            &q_norm_name,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let mut queries = operators.linear_rows(&self.payload.query_b, &q_latents, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qb,
            stage_start,
        )?;
        let stage_start = Instant::now();
        operators.rms_norm_heads_in_place(
            &mut queries,
            tokens * cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
            self.layer,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QHeadNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        for token in 0..tokens {
            let position = start_pos + token;
            let row_start = token * cfg.q_full_dim();
            apply_rotary_tail_scaled(
                &mut queries[row_start..row_start + cfg.q_full_dim()],
                cfg.num_heads,
                cfg.head_dim,
                cfg.rope_head_dim,
                position,
                rope,
                false,
            )?;
        }
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QRope,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let kv_rows = operators.linear_rows(&self.payload.key_value, hidden, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvProj,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let kv_norm_name = format!("kv_norm_L{}", self.layer);
        let mut window_values = operators.rms_norm_rows(
            &kv_rows,
            tokens,
            &self.payload.key_value_norm,
            cfg.norm_eps,
            &kv_norm_name,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        for token in 0..tokens {
            let position = start_pos + token;
            let row_start = token * cfg.head_dim;
            let kv = &mut window_values[row_start..row_start + cfg.head_dim];
            apply_rotary_tail_scaled(
                kv,
                1,
                cfg.head_dim,
                cfg.rope_head_dim,
                position,
                rope,
                false,
            )?;
            quantize_attention_kv_for_qat_in_place(kv, cfg.head_dim, cfg.rope_head_dim)?;
        }
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvRopeQuant,
            stage_start,
        )?;
        let stage_start = Instant::now();
        for token in 0..tokens {
            let position = start_pos + token;
            let row_start = token * cfg.head_dim;
            cache.window.append(
                position,
                &window_values[row_start..row_start + cfg.head_dim],
            )?;
        }
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvCacheAppend,
            stage_start,
        )?;

        if let Some(indexer) = compressed.indexer.as_ref() {
            let stage_start = Instant::now();
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
            record_attention_stage(
                operators,
                self.layer,
                DeepSeekV4AttentionProfileStage::IndexerCompress,
                stage_start,
            )?;
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

        let stage_start = Instant::now();
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
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::MainCompress,
            stage_start,
        )?;
        if main_compressed.len() % cfg.head_dim != 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} main prefill compressed length {} is not divisible by {}",
                self.layer,
                main_compressed.len(),
                cfg.head_dim
            )));
        }
        cache.compressed.extend_from_slice(&main_compressed);

        let stage_start = Instant::now();
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
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::TopkBuild,
            stage_start,
        )?;

        let main_compressed_len = main_compressed.len() / cfg.head_dim;
        let mut values = Vec::with_capacity((tokens + main_compressed_len) * cfg.head_dim);
        values.extend_from_slice(&window_values);
        values.extend_from_slice(&main_compressed);
        let stage_start = Instant::now();
        let context = operators.sparse_attention(
            &queries,
            &values,
            &topk,
            &self.payload.attention_sink,
            tokens,
            tokens + main_compressed_len,
            cfg.sparse_spec_with_topk(topk_cols),
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::SparseAttention,
            stage_start,
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
        match operators.backend() {
            DeepSeekV4OperatorBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    let stage_start = Instant::now();
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
                    }
                    record_attention_stage(
                        operators,
                        self.layer,
                        DeepSeekV4AttentionProfileStage::ContextRope,
                        stage_start,
                    )?;

                    let stage_start = Instant::now();
                    let latent = operators.grouped_output_a_rows_from_host(
                        &self.payload.output_a,
                        &context,
                        tokens,
                        cfg,
                        self.layer,
                    )?;
                    record_attention_stage(
                        operators,
                        self.layer,
                        DeepSeekV4AttentionProfileStage::OutputA,
                        stage_start,
                    )?;

                    let stage_start = Instant::now();
                    let projected = operators.linear_rows_from_device_to_host(
                        &self.payload.output_b,
                        &latent,
                        tokens,
                    )?;
                    record_attention_stage(
                        operators,
                        self.layer,
                        DeepSeekV4AttentionProfileStage::OutputB,
                        stage_start,
                    )?;
                    Ok(projected)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(Error::Model(
                        "DeepSeek-V4 CUDA attention projection requires cuda feature".into(),
                    ))
                }
            }
            DeepSeekV4OperatorBackend::Cpu => {
                let mut out = Vec::with_capacity(tokens * cfg.hidden_size);
                for token in 0..tokens {
                    let position = start_pos + token;
                    let row_start = token * cfg.q_full_dim();
                    let row = &mut context[row_start..row_start + cfg.q_full_dim()];
                    let stage_start = Instant::now();
                    apply_rotary_tail_scaled(
                        row,
                        cfg.num_heads,
                        cfg.head_dim,
                        cfg.rope_head_dim,
                        position,
                        cfg.rope_params(),
                        true,
                    )?;
                    record_attention_stage(
                        operators,
                        self.layer,
                        DeepSeekV4AttentionProfileStage::ContextRope,
                        stage_start,
                    )?;
                    let stage_start = Instant::now();
                    let latent =
                        operators.grouped_output_a(&self.payload.output_a, row, cfg, self.layer)?;
                    record_attention_stage(
                        operators,
                        self.layer,
                        DeepSeekV4AttentionProfileStage::OutputA,
                        stage_start,
                    )?;
                    let stage_start = Instant::now();
                    let projected = operators.linear_matvec(&self.payload.output_b, &latent)?;
                    record_attention_stage(
                        operators,
                        self.layer,
                        DeepSeekV4AttentionProfileStage::OutputB,
                        stage_start,
                    )?;
                    out.extend_from_slice(&projected);
                }
                Ok(out)
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn prefill_start_from_device(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        start_pos: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let cfg = self.config;
        if hidden_dev.len() == 0 || !hidden_dev.len().is_multiple_of(cfg.hidden_size) {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} device prefill hidden length mismatch: hidden={} dim={}",
                self.layer,
                hidden_dev.len(),
                cfg.hidden_size
            )));
        }
        if start_pos != 0 {
            return self.prefill_segment_from_device(cache, hidden_dev, start_pos, operators);
        }
        if cfg.compress_ratio == 0 {
            self.prefill_start_no_compress_from_device(cache, hidden_dev, start_pos, operators)
        } else {
            self.prefill_start_compressed_from_device(cache, hidden_dev, start_pos, operators)
        }
    }

    #[cfg(feature = "cuda")]
    fn prefill_segment_from_device(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        start_pos: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let cfg = self.config;
        let tokens = hidden_dev.len() / cfg.hidden_size;
        let mut output_dev = operators.cuda_zero_f32(hidden_dev.len())?;
        for token in 0..tokens {
            let row_dev =
                operators.gather_f32_rows(hidden_dev, &[token as i32], 1, cfg.hidden_size)?;
            let out_dev =
                self.decode_step_from_device(cache, &row_dev, start_pos + token, operators)?;
            operators.cuda_copy_f32_into_slot(
                &out_dev,
                &mut output_dev,
                token * cfg.hidden_size,
            )?;
        }
        Ok(output_dev)
    }

    #[cfg(feature = "cuda")]
    fn prefill_start_no_compress_from_device(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        start_pos: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let cfg = self.config;
        let tokens = hidden_dev.len() / cfg.hidden_size;
        operators.record_attention_call(self.layer, tokens);
        if cache.window.head_dim != cfg.head_dim || cache.window.window_size != cfg.window_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} KV cache shape mismatch: cache window={} head_dim={}, expected window={} head_dim={}",
                self.layer, cache.window.window_size, cache.window.head_dim, cfg.window_size, cfg.head_dim
            )));
        }
        let layer_tag = format!("attn_L{}", self.layer);

        let stage_start = Instant::now();
        let q_latents_dev =
            operators.linear_rows_from_device(&self.payload.query_a, hidden_dev, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qa,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let q_norm_name = format!("q_norm_{layer_tag}");
        let q_latents_dev = operators.cuda_rms_norm_rows_device_cached(
            &q_norm_name,
            &q_latents_dev,
            tokens,
            &self.payload.query_norm,
            cfg.norm_eps,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let queries_dev =
            operators.linear_rows_from_device(&self.payload.query_b, &q_latents_dev, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qb,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let mut queries_dev = operators.cuda_rms_norm_heads_from_device(
            &queries_dev,
            tokens * cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QHeadNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let rope_name = format!("rope_{layer_tag}");
        operators.cuda_ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            4096,
        )?;
        operators.cuda_rope_tail_rows_from_device(
            &rope_name,
            &mut queries_dev,
            start_pos as u32,
            tokens as u32,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QRope,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let kv_rows_dev =
            operators.linear_rows_from_device(&self.payload.key_value, hidden_dev, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvProj,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let kv_norm_name = format!("kv_norm_{layer_tag}");
        let mut values_dev = operators.cuda_rms_norm_rows_device_cached(
            &kv_norm_name,
            &kv_rows_dev,
            tokens,
            &self.payload.key_value_norm,
            cfg.norm_eps,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        operators.cuda_rope_tail_rows_from_device(
            &rope_name,
            &mut values_dev,
            start_pos as u32,
            tokens as u32,
            1,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        operators.cuda_attention_kv_qat_quantize_buffer_in_place(
            &mut values_dev,
            cfg.head_dim,
            cfg.rope_head_dim,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvRopeQuant,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let zero_kv = vec![0.0f32; cfg.head_dim];
        for token in 0..tokens {
            cache.window.append(start_pos + token, &zero_kv)?;
        }
        operators.cuda_ensure_kv_cache(self.layer, cfg.window_size, cfg.head_dim)?;
        operators.cuda_kv_write_window_rows_device(
            self.layer,
            &values_dev,
            start_pos,
            tokens,
            cfg.window_size,
            cfg.head_dim,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvCacheAppend,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let topk_cols = tokens.min(cfg.window_size);
        let topk_dev = operators.cuda_prefill_topk_indices_from_device(
            None,
            None,
            None,
            tokens,
            cfg.window_size,
            topk_cols,
            0,
            tokens,
            cfg.compress_ratio,
            0,
            0,
            0,
            1.0,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::TopkBuild,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let context = operators.cuda_sparse_attention_with_device_query_values_topk(
            &queries_dev,
            &values_dev,
            &topk_dev,
            &self.payload.attention_sink,
            tokens,
            tokens,
            cfg.sparse_spec_with_topk(topk_cols),
            self.layer,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::SparseAttention,
            stage_start,
        )?;
        self.project_context_rows_device_to_device(context, start_pos, tokens, operators)
    }

    #[cfg(feature = "cuda")]
    fn prefill_start_compressed_from_device(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        start_pos: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let cfg = self.config;
        let tokens = hidden_dev.len() / cfg.hidden_size;
        operators.record_attention_call(self.layer, tokens);
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
        let layer_tag = format!("attn_L{}", self.layer);

        let stage_start = Instant::now();
        let q_latents_dev =
            operators.linear_rows_from_device(&self.payload.query_a, hidden_dev, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qa,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let q_norm_name = format!("q_norm_{layer_tag}");
        let q_latents_dev = operators.cuda_rms_norm_rows_device_cached(
            &q_norm_name,
            &q_latents_dev,
            tokens,
            &self.payload.query_norm,
            cfg.norm_eps,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let queries_dev =
            operators.linear_rows_from_device(&self.payload.query_b, &q_latents_dev, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qb,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let mut queries_dev = operators.cuda_rms_norm_heads_from_device(
            &queries_dev,
            tokens * cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QHeadNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let rope_name = format!("rope_{layer_tag}");
        operators.cuda_ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            4096,
        )?;
        operators.cuda_rope_tail_rows_from_device(
            &rope_name,
            &mut queries_dev,
            start_pos as u32,
            tokens as u32,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QRope,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let kv_rows_dev =
            operators.linear_rows_from_device(&self.payload.key_value, hidden_dev, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvProj,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let kv_norm_name = format!("kv_norm_{layer_tag}");
        let mut window_values_dev = operators.cuda_rms_norm_rows_device_cached(
            &kv_norm_name,
            &kv_rows_dev,
            tokens,
            &self.payload.key_value_norm,
            cfg.norm_eps,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        operators.cuda_rope_tail_rows_from_device(
            &rope_name,
            &mut window_values_dev,
            start_pos as u32,
            tokens as u32,
            1,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        operators.cuda_attention_kv_qat_quantize_buffer_in_place(
            &mut window_values_dev,
            cfg.head_dim,
            cfg.rope_head_dim,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvRopeQuant,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let zero_kv = vec![0.0f32; cfg.head_dim];
        for token in 0..tokens {
            cache.window.append(start_pos + token, &zero_kv)?;
        }
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvCacheAppend,
            stage_start,
        )?;

        let mut indexer_values_dev: Option<ferrule_cuda::context::CudaF32Buffer> = None;
        if let Some(indexer) = compressed.indexer.as_ref() {
            let indexer_compressed_start = cache.indexer_compressed_len(cfg.index_head_dim);
            let stage_start = Instant::now();
            let (new_indexer_values_dev, indexer_values) = {
                let state = cache.indexer_compressor.as_mut().ok_or_else(|| {
                    Error::Model(format!(
                        "DeepSeek-V4 layer {} missing indexer compressor state",
                        self.layer
                    ))
                })?;
                state.prefill_start_device_output(
                    &indexer.compressor,
                    hidden_dev,
                    cfg.rope_head_dim,
                    rope,
                    &format!("rope_indexer_compress_L{}", self.layer),
                    &format!("indexer_compress_norm_L{}", self.layer),
                    operators,
                )?
            };
            record_attention_stage(
                operators,
                self.layer,
                DeepSeekV4AttentionProfileStage::IndexerCompress,
                stage_start,
            )?;
            if indexer_values.len() % cfg.index_head_dim != 0 {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 layer {} indexer prefill compressed length {} is not divisible by {}",
                    self.layer,
                    indexer_values.len(),
                    cfg.index_head_dim
                )));
            }
            cache.indexer_compressed.extend_from_slice(&indexer_values);
            let indexer_rows = new_indexer_values_dev.len() / cfg.index_head_dim;
            operators.cuda_ensure_indexer_kv_cache(
                self.layer,
                indexer_compressed_start + indexer_rows,
                cfg.index_head_dim,
            )?;
            operators.cuda_indexer_kv_write_rows_device(
                self.layer,
                &new_indexer_values_dev,
                indexer_compressed_start,
                cfg.index_head_dim,
            )?;
            indexer_values_dev = Some(new_indexer_values_dev);
        }

        let stage_start = Instant::now();
        let compressed_start = cache.compressed_len();
        let (main_compressed_dev, main_compressed) = {
            let state = cache.main_compressor.as_mut().ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 layer {} missing main compressor state",
                    self.layer
                ))
            })?;
            state.prefill_start_device_output(
                &compressed.compressor,
                hidden_dev,
                cfg.rope_head_dim,
                rope,
                &format!("rope_main_compress_L{}", self.layer),
                &format!("main_compress_norm_L{}", self.layer),
                operators,
            )?
        };
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::MainCompress,
            stage_start,
        )?;
        if main_compressed.len() % cfg.head_dim != 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} main prefill compressed length {} is not divisible by {}",
                self.layer,
                main_compressed.len(),
                cfg.head_dim
            )));
        }
        cache.compressed.extend_from_slice(&main_compressed);

        let stage_start = Instant::now();
        operators.cuda_ensure_combined_kv_cache(self.layer, cache, cache.compressed_len())?;
        operators.cuda_combined_kv_write_window_rows_device(
            self.layer,
            &window_values_dev,
            start_pos,
            tokens,
            cfg.window_size,
            cfg.head_dim,
        )?;
        operators.cuda_combined_kv_write_compressed_rows_device(
            self.layer,
            &main_compressed_dev,
            compressed_start,
            cfg.window_size,
            cfg.head_dim,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::CompressedKvUpload,
            stage_start,
        )?;

        let main_compressed_len = main_compressed_dev.len() / cfg.head_dim;
        let stage_start = Instant::now();
        let window_cols = tokens.min(cfg.window_size);
        let compressed_offset = tokens;
        let (topk_dev, topk_cols) = if let Some(indexer) = compressed.indexer.as_ref() {
            let indexer_kv_dev = indexer_values_dev.as_ref().ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 layer {} missing device indexer compressed values for prefill topk",
                    self.layer
                ))
            })?;
            let compressed_len = indexer_kv_dev.len() / cfg.index_head_dim;
            let cached_compressed_len = cache.indexer_compressed_len(cfg.index_head_dim);
            if compressed_len != cached_compressed_len {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 layer {} device indexer compressed length mismatch: device={compressed_len} cache={cached_compressed_len}",
                    self.layer
                )));
            }
            let indexer_cols = cfg.index_topk.min(compressed_len);
            if indexer_cols == 0 {
                let topk_dev = operators.cuda_prefill_topk_indices_from_device(
                    None,
                    None,
                    None,
                    tokens,
                    cfg.window_size,
                    window_cols,
                    0,
                    compressed_offset,
                    cfg.compress_ratio,
                    0,
                    0,
                    0,
                    1.0,
                )?;
                (topk_dev, window_cols)
            } else {
                let mut index_query_dev =
                    operators.linear_rows_from_device(&indexer.wq_b, &q_latents_dev, tokens)?;
                let index_rope_dim = cfg.rope_head_dim.min(cfg.index_head_dim);
                let index_rope_name = format!("rope_indexer_query_L{}", self.layer);
                operators.cuda_ensure_rope_tables_with_params(
                    &index_rope_name,
                    index_rope_dim,
                    cfg.rope_params(),
                    4096,
                )?;
                let index_weights_dev =
                    operators.linear_rows_from_device(&indexer.weights_proj, hidden_dev, tokens)?;
                let weight_scale =
                    (cfg.index_head_dim as f32).powf(-0.5) * (cfg.index_n_heads as f32).powf(-0.5);
                let topk_dev = if dsv4_fused_indexer_topk_supported(cfg, index_rope_dim) {
                    operators.cuda_prefill_topk_indices_fused_index_query_from_device(
                        &index_query_dev,
                        &index_weights_dev,
                        indexer_kv_dev,
                        &index_rope_name,
                        tokens,
                        cfg.window_size,
                        window_cols,
                        indexer_cols,
                        compressed_offset,
                        cfg.compress_ratio,
                        compressed_len,
                        cfg.index_n_heads,
                        cfg.index_head_dim,
                        index_rope_dim,
                        start_pos,
                        weight_scale,
                    )?
                } else {
                    operators.cuda_rope_tail_rows_from_device(
                        &index_rope_name,
                        &mut index_query_dev,
                        start_pos as u32,
                        tokens as u32,
                        cfg.index_n_heads as u32,
                        cfg.index_head_dim as u32,
                        index_rope_dim as u32,
                        false,
                    )?;
                    operators.cuda_indexer_qat_quantize_buffer_in_place(
                        &mut index_query_dev,
                        cfg.index_head_dim,
                    )?;
                    operators.cuda_prefill_topk_indices_from_device(
                        Some(&index_query_dev),
                        Some(&index_weights_dev),
                        Some(indexer_kv_dev),
                        tokens,
                        cfg.window_size,
                        window_cols,
                        indexer_cols,
                        compressed_offset,
                        cfg.compress_ratio,
                        compressed_len,
                        cfg.index_n_heads,
                        cfg.index_head_dim,
                        weight_scale,
                    )?
                };
                (topk_dev, window_cols + indexer_cols)
            }
        } else {
            let compressed_cols = main_compressed_len;
            let topk_dev = operators.cuda_prefill_topk_indices_from_device(
                None,
                None,
                None,
                tokens,
                cfg.window_size,
                window_cols,
                compressed_cols,
                compressed_offset,
                cfg.compress_ratio,
                main_compressed_len,
                0,
                0,
                1.0,
            )?;
            (topk_dev, window_cols + compressed_cols)
        };
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::TopkBuild,
            stage_start,
        )?;

        let compact_values = operators.concat_attention_values_device_buffers(
            &window_values_dev,
            &main_compressed_dev,
            tokens,
            cfg.head_dim,
        )?;
        let stage_start = Instant::now();
        let context = operators.cuda_sparse_attention_with_device_query_values_topk(
            &queries_dev,
            &compact_values,
            &topk_dev,
            &self.payload.attention_sink,
            tokens,
            tokens + main_compressed_len,
            cfg.sparse_spec_with_topk(topk_cols),
            self.layer,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::SparseAttention,
            stage_start,
        )?;
        self.project_context_rows_device_to_device(context, start_pos, tokens, operators)
    }

    #[cfg(feature = "cuda")]
    fn project_context_rows_device_to_device(
        &self,
        mut context: ferrule_cuda::context::CudaF32Buffer,
        start_pos: usize,
        tokens: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let cfg = self.config;
        if context.len() != tokens * cfg.q_full_dim() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} device attention context length mismatch: expected {}, got {}",
                self.layer,
                tokens * cfg.q_full_dim(),
                context.len()
            )));
        }
        let stage_start = Instant::now();
        let rope_name = format!("rope_attn_L{}", self.layer);
        operators.cuda_ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            4096,
        )?;
        operators.cuda_rope_tail_rows_from_device(
            &rope_name,
            &mut context,
            start_pos as u32,
            tokens as u32,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            true,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::ContextRope,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let latent = operators.grouped_output_a_rows_from_device(
            &self.payload.output_a,
            &context,
            tokens,
            cfg,
            self.layer,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::OutputA,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let output = operators.linear_rows_from_device(&self.payload.output_b, &latent, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::OutputB,
            stage_start,
        )?;
        Ok(output)
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

        #[cfg(feature = "cuda")]
        if operators.backend == DeepSeekV4OperatorBackend::Cuda {
            return self.decode_step_compressed_device(
                cache, hidden, position, operators, cfg, compressed, rope,
            );
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
        quantize_attention_kv_for_qat_in_place(&mut kv, cfg.head_dim, cfg.rope_head_dim)?;
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

    /// Device-resident attention decode: accepts hidden as `CudaF32Buffer`,
    /// returns `CudaF32Buffer`. Eliminates the D2H+H2D round-trip at the
    /// call boundary that `decode_step_with_operators` forces.
    #[cfg(feature = "cuda")]
    pub fn decode_step_from_device(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let cfg = self.config;
        if cfg.compress_ratio == 0 {
            return self.decode_step_no_compress_from_device(
                &mut cache.window,
                hidden_dev,
                position,
                operators,
                cfg,
            );
        }
        if hidden_dev.len() != cfg.hidden_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} attention device input mismatch: expected {}, got {}",
                self.layer,
                cfg.hidden_size,
                hidden_dev.len()
            )));
        }
        let compressed = self.compressed.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {} has compress_ratio {} but no typed compressed attention payload is bound",
                self.layer, cfg.compress_ratio
            ))
        })?;
        let rope = cfg.rope_params();
        self.decode_step_compressed_from_device(
            cache, hidden_dev, position, operators, cfg, compressed, rope,
        )
    }

    /// Graph-safe attention decode over already-prepared CUDA KV caches.
    ///
    /// The eager warmup pass updates window/compressed/indexer KV and materializes
    /// combined KV caches. CUDA graph capture then records the fixed-shape query,
    /// top-k, sparse attention and output projection kernels without compressor
    /// host-shadow downloads or cache-growth allocations.
    #[cfg(feature = "cuda")]
    pub(crate) fn decode_step_graph_safe_from_prepared_cache(
        &self,
        cache: &DeepSeekV4AttentionCache,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        operators: &mut DeepSeekV4OperatorContext,
        arena: &mut DeepSeekV4AttentionGraphArena,
    ) -> Result<()> {
        let cfg = self.config;
        if hidden_dev.len() != cfg.hidden_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} graph-safe attention input mismatch: expected {}, got {}",
                self.layer,
                cfg.hidden_size,
                hidden_dev.len()
            )));
        }
        if !arena.is_compatible(cfg) {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} graph-safe attention arena shape is stale",
                self.layer
            )));
        }
        let compressed = if cfg.compress_ratio == 0 {
            None
        } else {
            Some(self.compressed.as_ref().ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 layer {} has compress_ratio {} but no typed compressed attention payload is bound",
                    self.layer, cfg.compress_ratio
                ))
            })?)
        };
        let layer_tag = format!("attn_L{}", self.layer);
        let rope_name = format!("rope_{layer_tag}");
        operators.cuda_ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            4096,
        )?;

        operators.cuda_copy_f32_into_slot(hidden_dev, &mut arena.hidden_q, 0)?;
        operators.cuda_linear_matvec_from_device_into(
            &self.payload.query_a,
            &mut arena.hidden_q,
            &mut arena.q_latent,
        )?;
        let q_norm_name = format!("q_norm_{layer_tag}");
        operators.cuda_rms_norm_device_cached_into(
            &q_norm_name,
            &arena.q_latent,
            &self.payload.query_norm,
            cfg.norm_eps,
            &mut arena.q_normed,
        )?;
        if compressed
            .and_then(|payload| payload.indexer.as_ref())
            .is_some()
        {
            operators.cuda_copy_f32_into_slot(
                &arena.q_normed,
                &mut arena.q_normed_for_indexer,
                0,
            )?;
        }
        operators.cuda_linear_matvec_from_device_into(
            &self.payload.query_b,
            &mut arena.q_normed,
            &mut arena.query,
        )?;
        operators.cuda_rms_norm_heads_from_device_into(
            &arena.query,
            cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
            &mut arena.query_normed,
        )?;
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut arena.query_normed,
            position as u32,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;

        // The capture warmup has already advanced the host-side KV metadata. During
        // graph replay we still must overwrite the current token's device KV slot
        // from the graph-computed hidden. Otherwise deeper layers mix a replayed
        // query with a warmup-produced current-token KV row and diverge.
        operators.cuda_copy_f32_into_slot(hidden_dev, &mut arena.hidden_q, 0)?;
        operators.cuda_linear_matvec_from_device_into(
            &self.payload.key_value,
            &mut arena.hidden_q,
            &mut arena.kv,
        )?;
        let kv_norm_name = format!("kv_norm_{layer_tag}");
        operators.cuda_rms_norm_device_cached_into(
            &kv_norm_name,
            &arena.kv,
            &self.payload.key_value_norm,
            cfg.norm_eps,
            &mut arena.kv_normed,
        )?;
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut arena.kv_normed,
            position as u32,
            1,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        operators.cuda_attention_kv_qat_quantize_buffer_in_place(
            &mut arena.kv_normed,
            cfg.head_dim,
            cfg.rope_head_dim,
        )?;
        if compressed.is_some() {
            operators.cuda_combined_kv_append_window_device(
                self.layer,
                &arena.kv_normed,
                position,
                cfg.window_size,
                cfg.head_dim,
            )?;
        } else {
            operators.cuda_kv_write_window_device(
                self.layer,
                &arena.kv_normed,
                position,
                cfg.head_dim,
                cfg.window_size,
            )?;
        }

        let window_len = cache.window.len;
        let attention_kv_len = if compressed.is_some() {
            cache.kv_len_for_attention()
        } else {
            cfg.window_size
        };
        let topk_len = if let Some(indexer) =
            compressed.and_then(|payload| payload.indexer.as_ref())
        {
            let compressed_len = cache.indexer_compressed_len(cfg.index_head_dim);
            let indexer_cols = cfg.index_topk.min(compressed_len);
            if indexer_cols == 0 {
                operators.cuda_decode_topk_indices_from_device_into(
                    None,
                    None,
                    None,
                    &arena.empty_query,
                    &arena.empty_weights,
                    &arena.empty_kv,
                    position,
                    window_len,
                    cfg.window_size,
                    0,
                    cfg.window_size,
                    0,
                    0,
                    0,
                    1.0,
                    &mut arena.topk,
                )?;
            } else {
                operators.cuda_linear_matvec_from_device_into(
                    &indexer.wq_b,
                    &mut arena.q_normed_for_indexer,
                    &mut arena.index_query,
                )?;
                let index_rope_dim = cfg.rope_head_dim.min(cfg.index_head_dim);
                let index_rope_name = format!("rope_indexer_query_L{}", self.layer);
                operators.cuda_ensure_rope_tables_with_params(
                    &index_rope_name,
                    index_rope_dim,
                    cfg.rope_params(),
                    4096,
                )?;
                let weight_scale =
                    (cfg.index_head_dim as f32).powf(-0.5) * (cfg.index_n_heads as f32).powf(-0.5);
                if dsv4_fused_indexer_decode_topk_supported(cfg, index_rope_dim) {
                    operators.cuda_copy_f32_into_slot(
                        hidden_dev,
                        &mut arena.index_weight_input,
                        0,
                    )?;
                    operators.cuda_linear_matvec_from_device_into(
                        &indexer.weights_proj,
                        &mut arena.index_weight_input,
                        &mut arena.index_weights,
                    )?;
                    operators.cuda_decode_topk_indices_fused_index_query_from_indexer_cache_into(
                        &arena.index_query,
                        &arena.index_weights,
                        self.layer,
                        &index_rope_name,
                        position,
                        window_len,
                        cfg.window_size,
                        indexer_cols,
                        cfg.window_size,
                        compressed_len,
                        cfg.index_n_heads,
                        cfg.index_head_dim,
                        index_rope_dim,
                        weight_scale,
                        &mut arena.topk,
                    )?;
                } else {
                    operators.cuda_rope_tail_from_device(
                        &index_rope_name,
                        &mut arena.index_query,
                        position as u32,
                        cfg.index_n_heads as u32,
                        cfg.index_head_dim as u32,
                        index_rope_dim as u32,
                        false,
                    )?;
                    operators.cuda_indexer_qat_quantize_buffer_in_place(
                        &mut arena.index_query,
                        cfg.index_head_dim,
                    )?;
                    operators.cuda_copy_f32_into_slot(
                        hidden_dev,
                        &mut arena.index_weight_input,
                        0,
                    )?;
                    operators.cuda_linear_matvec_from_device_into(
                        &indexer.weights_proj,
                        &mut arena.index_weight_input,
                        &mut arena.index_weights,
                    )?;
                    operators.cuda_decode_topk_indices_from_indexer_cache_into(
                        &arena.index_query,
                        &arena.index_weights,
                        self.layer,
                        &arena.empty_query,
                        &arena.empty_weights,
                        &arena.empty_kv,
                        position,
                        window_len,
                        cfg.window_size,
                        indexer_cols,
                        cfg.window_size,
                        compressed_len,
                        cfg.index_n_heads,
                        cfg.index_head_dim,
                        weight_scale,
                        &mut arena.topk,
                    )?;
                }
            }
            cfg.window_size + indexer_cols
        } else {
            let compressed_len = cache.compressed_len();
            let topk_len = cfg.window_size + compressed_len;
            if topk_len > arena.max_topk {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 layer {} graph-safe no-indexer topk {} exceeds arena capacity {}",
                    self.layer, topk_len, arena.max_topk
                )));
            }
            operators.cuda_decode_topk_indices_from_device_into(
                None,
                None,
                None,
                &arena.empty_query,
                &arena.empty_weights,
                &arena.empty_kv,
                position,
                window_len,
                cfg.window_size,
                compressed_len,
                cfg.window_size,
                compressed_len,
                0,
                0,
                1.0,
                &mut arena.topk,
            )?;
            topk_len
        };

        if compressed.is_some() {
            operators.cuda_sparse_attention_with_combined_kv_topk_into(
                &arena.query_normed,
                self.layer,
                &arena.topk,
                &self.payload.attention_sink,
                1,
                attention_kv_len,
                cfg.sparse_spec_with_topk(topk_len),
                &mut arena.context,
            )?;
        } else {
            operators.cuda_sparse_attention_with_kv_cache_topk_into(
                &arena.query_normed,
                self.layer,
                &arena.topk,
                &self.payload.attention_sink,
                1,
                attention_kv_len,
                cfg.sparse_spec_with_topk(topk_len),
                &mut arena.context,
            )?;
        }
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut arena.context,
            position as u32,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            true,
        )?;
        operators.grouped_output_a_from_device_into(
            &self.payload.output_a,
            &arena.context,
            cfg,
            self.layer,
            &mut arena.latent,
        )?;
        operators.cuda_linear_matvec_from_device_into(
            &self.payload.output_b,
            &mut arena.latent,
            &mut arena.output,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn decode_step_compressed_device(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden: &[f32],
        position: usize,
        operators: &mut DeepSeekV4OperatorContext,
        cfg: DeepSeekV4AttentionConfig,
        compressed: &DeepSeekV4CompressedAttentionPayload,
        rope: DeepSeekV4RopeParams,
    ) -> Result<Vec<f32>> {
        let layer_tag = format!("attn_L{}", self.layer);
        let rope_name = format!("rope_{layer_tag}");
        operators.cuda_ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            4096,
        )?;
        let mut hidden_device = operators.cuda_upload_f32(hidden)?;
        let mut kv_hidden_device = operators.cuda_clone_f32(&hidden_device)?;

        let q_latent_dev =
            operators.cuda_linear_matvec_from_device(&self.payload.query_a, &mut hidden_device)?;
        let q_norm_name = format!("q_norm_{layer_tag}");
        let q_normed_dev = operators.cuda_rms_norm_device_cached(
            &q_norm_name,
            &q_latent_dev,
            &self.payload.query_norm,
            cfg.norm_eps,
        )?;
        // Indexer routing still uses the normalized q-latent on host.
        // Only download if indexer exists (avoid unnecessary D2H sync).
        let q_latent = if compressed.indexer.is_some() {
            operators.cuda_download_f32(&q_normed_dev)?
        } else {
            Vec::new()
        };
        let mut q_normed_dev = q_normed_dev;
        let mut query =
            operators.cuda_linear_matvec_from_device(&self.payload.query_b, &mut q_normed_dev)?;
        query = operators.cuda_rms_norm_heads_from_device(
            &query,
            cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
        )?;
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut query,
            position as u32,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;

        let kv_dev = operators
            .cuda_linear_matvec_from_device(&self.payload.key_value, &mut kv_hidden_device)?;
        let kv_norm_name = format!("kv_norm_{layer_tag}");
        let mut kv_normed_dev = operators.cuda_rms_norm_device_cached(
            &kv_norm_name,
            &kv_dev,
            &self.payload.key_value_norm,
            cfg.norm_eps,
        )?;
        // Device-side rotary + QAT: no D2H needed.
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut kv_normed_dev,
            position as u32,
            1,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        operators.cuda_attention_kv_qat_quantize_buffer_in_place(
            &mut kv_normed_dev,
            cfg.head_dim,
            cfg.rope_head_dim,
        )?;
        // Append to device combined KV cache (no D2H/H2D round-trip).
        operators.cuda_ensure_combined_kv_cache(self.layer, cache, cache.compressed_len() + 1)?;
        operators.cuda_combined_kv_append_window_device(
            self.layer,
            &kv_normed_dev,
            position,
            cfg.window_size,
            cfg.head_dim,
        )?;
        // Still need host-side cache.window.append for topk_indices,
        // but use a zero vector — only len tracking matters for topk.
        cache.window.append(position, &vec![0.0f32; cfg.head_dim])?;

        if let Some(indexer) = compressed.indexer.as_ref() {
            let new_indexer_kv = {
                let state = cache.indexer_compressor.as_mut().ok_or_else(|| {
                    Error::Model(format!(
                        "DeepSeek-V4 layer {} missing indexer compressor state",
                        self.layer
                    ))
                })?;
                if indexer
                    .compressor
                    .wkv
                    .execution
                    .activation_quantization
                    .is_none()
                    && indexer
                        .compressor
                        .wgate
                        .execution
                        .activation_quantization
                        .is_none()
                {
                    let mut compressor_hidden_device = operators.cuda_upload_f32(hidden)?;
                    state.append_step_from_device_if_unquantized(
                        &indexer.compressor,
                        &mut compressor_hidden_device,
                        position,
                        cfg.rope_head_dim,
                        rope,
                        operators,
                    )?
                } else {
                    state.append_step(
                        &indexer.compressor,
                        hidden,
                        position,
                        cfg.rope_head_dim,
                        rope,
                        operators,
                    )?
                }
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
            if compressed
                .compressor
                .wkv
                .execution
                .activation_quantization
                .is_none()
                && compressed
                    .compressor
                    .wgate
                    .execution
                    .activation_quantization
                    .is_none()
            {
                let mut compressor_hidden_device = operators.cuda_upload_f32(hidden)?;
                state.append_step_from_device_if_unquantized(
                    &compressed.compressor,
                    &mut compressor_hidden_device,
                    position,
                    cfg.rope_head_dim,
                    rope,
                    operators,
                )?
            } else {
                state.append_step(
                    &compressed.compressor,
                    hidden,
                    position,
                    cfg.rope_head_dim,
                    rope,
                    operators,
                )?
            }
        };
        if let Some(value) = new_main_kv.as_ref() {
            cache.append_compressed(value)?;
        }

        // Combined KV cache window append already done above (device path).
        // Only compressed KV append remains here.
        if let Some(value) = new_main_kv.as_ref() {
            operators.cuda_ensure_combined_kv_cache(
                self.layer,
                cache,
                cache.compressed_len() + 1,
            )?;
            let compressed_index = cache.compressed_len().saturating_sub(1);
            operators.cuda_combined_kv_append_compressed_host(
                self.layer,
                value,
                compressed_index,
                cfg.window_size,
                cfg.head_dim,
            )?;
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

        let mut context = operators.cuda_sparse_attention_with_combined_kv(
            &query,
            self.layer,
            &topk,
            &self.payload.attention_sink,
            1,
            cache.kv_len_for_attention(),
            cfg.sparse_spec_with_topk(topk.len()),
        )?;
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut context,
            position as u32,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            true,
        )?;
        let latent = operators.grouped_output_a_from_device(
            &self.payload.output_a,
            &context,
            cfg,
            self.layer,
        )?;
        let mut latent = latent;
        let output =
            operators.cuda_linear_matvec_from_device(&self.payload.output_b, &mut latent)?;
        operators.cuda_download_f32(&output)
    }

    /// CUDA fully device-resident compressed attention path.
    ///
    /// Like `decode_step_compressed_device` but accepts the hidden state as a
    /// `CudaF32Buffer` and returns the output as a `CudaF32Buffer`, eliminating
    /// the D2H+H2D round-trip at the call boundary (4 syncs → 0 in steady state).
    ///
    /// Remaining host syncs are now limited to compressor CPU-shadow state
    /// maintenance and MoE/router bookkeeping outside this attention function.
    /// q-latent/hidden downloads and top-k uploads have been removed from this
    /// decode path; indexer compressed KV is served from a CUDA-side cache.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn decode_step_compressed_from_device(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        operators: &mut DeepSeekV4OperatorContext,
        cfg: DeepSeekV4AttentionConfig,
        compressed: &DeepSeekV4CompressedAttentionPayload,
        rope: DeepSeekV4RopeParams,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let layer_tag = format!("attn_L{}", self.layer);
        let rope_name = format!("rope_{layer_tag}");
        operators.record_attention_call(self.layer, 1);
        operators.cuda_ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            4096,
        )?;
        // Clone the hidden device buffer so each activation-quantized linear sees
        // the original input; `cuda_linear_matvec_from_device` mutates its input.
        let mut hidden_device = operators.cuda_clone_f32(hidden_dev)?;
        let mut kv_hidden_device = operators.cuda_clone_f32(hidden_dev)?;

        let stage_start = Instant::now();
        let q_latent_dev =
            operators.cuda_linear_matvec_from_device(&self.payload.query_a, &mut hidden_device)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qa,
            stage_start,
        )?;
        let q_norm_name = format!("q_norm_{layer_tag}");
        let stage_start = Instant::now();
        let q_normed_dev = operators.cuda_rms_norm_device_cached(
            &q_norm_name,
            &q_latent_dev,
            &self.payload.query_norm,
            cfg.norm_eps,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QNorm,
            stage_start,
        )?;
        let q_normed_for_indexer = if compressed.indexer.is_some() {
            Some(operators.cuda_clone_f32(&q_normed_dev)?)
        } else {
            None
        };
        let mut q_normed_dev = q_normed_dev;
        let stage_start = Instant::now();
        let mut query =
            operators.cuda_linear_matvec_from_device(&self.payload.query_b, &mut q_normed_dev)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qb,
            stage_start,
        )?;
        let stage_start = Instant::now();
        query = operators.cuda_rms_norm_heads_from_device(
            &query,
            cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QHeadNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut query,
            position as u32,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QRope,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let kv_dev = operators
            .cuda_linear_matvec_from_device(&self.payload.key_value, &mut kv_hidden_device)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvProj,
            stage_start,
        )?;
        let kv_norm_name = format!("kv_norm_{layer_tag}");
        let stage_start = Instant::now();
        let mut kv_normed_dev = operators.cuda_rms_norm_device_cached(
            &kv_norm_name,
            &kv_dev,
            &self.payload.key_value_norm,
            cfg.norm_eps,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvNorm,
            stage_start,
        )?;
        // Device-side rotary + QAT: no D2H needed.
        let stage_start = Instant::now();
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut kv_normed_dev,
            position as u32,
            1,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        operators.cuda_attention_kv_qat_quantize_buffer_in_place(
            &mut kv_normed_dev,
            cfg.head_dim,
            cfg.rope_head_dim,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvRopeQuant,
            stage_start,
        )?;
        // Append to device combined KV cache (no D2H/H2D round-trip).
        let stage_start = Instant::now();
        operators.cuda_ensure_combined_kv_cache(self.layer, cache, cache.compressed_len() + 1)?;
        operators.cuda_combined_kv_append_window_device(
            self.layer,
            &kv_normed_dev,
            position,
            cfg.window_size,
            cfg.head_dim,
        )?;
        // Still need host-side cache.window.append for topk_indices,
        // but use a zero vector — only len tracking matters for topk.
        cache.window.append(position, &vec![0.0f32; cfg.head_dim])?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvCacheAppend,
            stage_start,
        )?;

        if let Some(indexer) = compressed.indexer.as_ref() {
            let stage_start = Instant::now();
            let new_indexer_kv = {
                let state = cache.indexer_compressor.as_mut().ok_or_else(|| {
                    Error::Model(format!(
                        "DeepSeek-V4 layer {} missing indexer compressor state",
                        self.layer
                    ))
                })?;
                state.append_step_from_device(
                    &indexer.compressor,
                    hidden_dev,
                    position,
                    cfg.rope_head_dim,
                    rope,
                    operators,
                )?
            };
            record_attention_stage(
                operators,
                self.layer,
                DeepSeekV4AttentionProfileStage::IndexerCompress,
                stage_start,
            )?;
            if let Some(value) = new_indexer_kv {
                let compressed_index = cache.indexer_compressed_len(cfg.index_head_dim);
                cache.append_indexer_compressed(&value, cfg.index_head_dim)?;
                operators.cuda_ensure_indexer_kv_cache(
                    self.layer,
                    cache.indexer_compressed_len(cfg.index_head_dim),
                    cfg.index_head_dim,
                )?;
                operators.cuda_indexer_kv_append_host(
                    self.layer,
                    &value,
                    compressed_index,
                    cfg.index_head_dim,
                )?;
            }
        }

        let stage_start = Instant::now();
        let new_main_kv = {
            let state = cache.main_compressor.as_mut().ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 layer {} missing main compressor state",
                    self.layer
                ))
            })?;
            state.append_step_from_device(
                &compressed.compressor,
                hidden_dev,
                position,
                cfg.rope_head_dim,
                rope,
                operators,
            )?
        };
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::MainCompress,
            stage_start,
        )?;
        if let Some(value) = new_main_kv.as_ref() {
            cache.append_compressed(value)?;
        }

        // Combined KV cache window append already done above (device path).
        // Only compressed KV append remains here.
        if let Some(value) = new_main_kv.as_ref() {
            let stage_start = Instant::now();
            operators.cuda_ensure_combined_kv_cache(
                self.layer,
                cache,
                cache.compressed_len() + 1,
            )?;
            let compressed_index = cache.compressed_len().saturating_sub(1);
            operators.cuda_combined_kv_append_compressed_host(
                self.layer,
                value,
                compressed_index,
                cfg.window_size,
                cfg.head_dim,
            )?;
            record_attention_stage(
                operators,
                self.layer,
                DeepSeekV4AttentionProfileStage::CompressedKvUpload,
                stage_start,
            )?;
        }

        let stage_start = Instant::now();
        let window_len = cache.window.len;
        let (topk_dev, topk_len) = if let Some(indexer) = compressed.indexer.as_ref() {
            let compressed_len = cache.indexer_compressed_len(cfg.index_head_dim);
            let indexer_cols = cfg.index_topk.min(compressed_len);
            if indexer_cols == 0 {
                let topk_dev = operators.cuda_decode_topk_indices_from_device(
                    None,
                    None,
                    None,
                    position,
                    window_len,
                    cfg.window_size,
                    0,
                    cfg.window_size,
                    0,
                    0,
                    0,
                    1.0,
                )?;
                (topk_dev, cfg.window_size)
            } else {
                let mut index_q_input = q_normed_for_indexer.ok_or_else(|| {
                    Error::Model(format!(
                        "DeepSeek-V4 layer {} missing device q-latent clone for decode indexer topk",
                        self.layer
                    ))
                })?;
                let mut index_query_dev =
                    operators.cuda_linear_matvec_from_device(&indexer.wq_b, &mut index_q_input)?;
                let index_rope_dim = cfg.rope_head_dim.min(cfg.index_head_dim);
                let index_rope_name = format!("rope_indexer_query_L{}", self.layer);
                operators.cuda_ensure_rope_tables_with_params(
                    &index_rope_name,
                    index_rope_dim,
                    cfg.rope_params(),
                    4096,
                )?;
                operators.cuda_ensure_indexer_kv_cache(
                    self.layer,
                    compressed_len,
                    cfg.index_head_dim,
                )?;
                let weight_scale =
                    (cfg.index_head_dim as f32).powf(-0.5) * (cfg.index_n_heads as f32).powf(-0.5);
                let topk_dev = if dsv4_fused_indexer_decode_topk_supported(cfg, index_rope_dim) {
                    let mut index_weight_input = operators.cuda_clone_f32(hidden_dev)?;
                    let index_weights_dev = operators.cuda_linear_matvec_from_device(
                        &indexer.weights_proj,
                        &mut index_weight_input,
                    )?;
                    operators.cuda_decode_topk_indices_fused_index_query_from_indexer_cache(
                        &index_query_dev,
                        &index_weights_dev,
                        self.layer,
                        &index_rope_name,
                        position,
                        window_len,
                        cfg.window_size,
                        indexer_cols,
                        cfg.window_size,
                        compressed_len,
                        cfg.index_n_heads,
                        cfg.index_head_dim,
                        index_rope_dim,
                        weight_scale,
                    )?
                } else {
                    operators.cuda_rope_tail_from_device(
                        &index_rope_name,
                        &mut index_query_dev,
                        position as u32,
                        cfg.index_n_heads as u32,
                        cfg.index_head_dim as u32,
                        index_rope_dim as u32,
                        false,
                    )?;
                    operators.cuda_indexer_qat_quantize_buffer_in_place(
                        &mut index_query_dev,
                        cfg.index_head_dim,
                    )?;
                    let mut index_weight_input = operators.cuda_clone_f32(hidden_dev)?;
                    let index_weights_dev = operators.cuda_linear_matvec_from_device(
                        &indexer.weights_proj,
                        &mut index_weight_input,
                    )?;
                    operators.cuda_decode_topk_indices_from_indexer_cache(
                        &index_query_dev,
                        &index_weights_dev,
                        self.layer,
                        position,
                        window_len,
                        cfg.window_size,
                        indexer_cols,
                        cfg.window_size,
                        compressed_len,
                        cfg.index_n_heads,
                        cfg.index_head_dim,
                        weight_scale,
                    )?
                };
                (topk_dev, cfg.window_size + indexer_cols)
            }
        } else {
            let compressed_len = cache.compressed_len();
            let topk_dev = operators.cuda_decode_topk_indices_from_device(
                None,
                None,
                None,
                position,
                window_len,
                cfg.window_size,
                compressed_len,
                cfg.window_size,
                compressed_len,
                0,
                0,
                1.0,
            )?;
            (topk_dev, cfg.window_size + compressed_len)
        };
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::TopkBuild,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let mut context = operators.cuda_sparse_attention_with_combined_kv_topk(
            &query,
            self.layer,
            &topk_dev,
            &self.payload.attention_sink,
            1,
            cache.kv_len_for_attention(),
            cfg.sparse_spec_with_topk(topk_len),
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::SparseAttention,
            stage_start,
        )?;
        let stage_start = Instant::now();
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut context,
            position as u32,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            true,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::ContextRope,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let latent = operators.grouped_output_a_from_device(
            &self.payload.output_a,
            &context,
            cfg,
            self.layer,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::OutputA,
            stage_start,
        )?;
        let mut latent = latent;
        let stage_start = Instant::now();
        let output =
            operators.cuda_linear_matvec_from_device(&self.payload.output_b, &mut latent)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::OutputB,
            stage_start,
        )?;
        // Return device buffer — no D2H download.
        Ok(output)
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

        // Device-resident path: chain linear_matvec + rms_norm on device.
        #[cfg(feature = "cuda")]
        if operators.backend == DeepSeekV4OperatorBackend::Cuda {
            return self.decode_step_no_compress_device(cache, hidden, position, operators, cfg);
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
        quantize_attention_kv_for_qat_in_place(&mut kv, cfg.head_dim, cfg.rope_head_dim)?;
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

    /// CUDA fully device-resident no-compress attention path.
    ///
    /// All operations stay on device: linear projections, rms_norm, rotary,
    /// KV cache append, sparse attention, grouped output_a, output_b.
    /// No D2H copies in steady state — enables CUDA graph capture.
    #[cfg(feature = "cuda")]
    fn decode_step_no_compress_device(
        &self,
        _cache: &mut DeepSeekV4WindowKvCache,
        hidden: &[f32],
        position: usize,
        operators: &mut DeepSeekV4OperatorContext,
        cfg: DeepSeekV4AttentionConfig,
    ) -> Result<Vec<f32>> {
        let layer_tag = format!("attn_L{}", self.layer);
        let rope_name = format!("rope_{layer_tag}");

        // Precompute rope tables (cached, uploaded once).
        operators.cuda_ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            4096,
        )?;

        // Keep Q and KV projection inputs independent because device matvec
        // applies activation quantization in-place.
        let mut hidden_device = operators.cuda_upload_f32(hidden)?;
        let mut kv_hidden_device = operators.cuda_clone_f32(&hidden_device)?;

        // Query: query_a → rms_norm → query_b → head_norm (all on device)
        let q_latent_dev =
            operators.cuda_linear_matvec_from_device(&self.payload.query_a, &mut hidden_device)?;
        let q_norm_name = format!("q_norm_{layer_tag}");
        let q_normed_dev = operators.cuda_rms_norm_device_cached(
            &q_norm_name,
            &q_latent_dev,
            &self.payload.query_norm,
            cfg.norm_eps,
        )?;
        let mut q_normed_dev = q_normed_dev;
        let mut query =
            operators.cuda_linear_matvec_from_device(&self.payload.query_b, &mut q_normed_dev)?;
        query = operators.cuda_rms_norm_heads_from_device(
            &query,
            cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
        )?;
        // Rotary on device.
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut query,
            position as u32,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;

        // KV: key_value → rms_norm → rotary (all on device), then append to device KV cache.
        let kv_dev = operators
            .cuda_linear_matvec_from_device(&self.payload.key_value, &mut kv_hidden_device)?;
        let kv_norm_name = format!("kv_norm_{layer_tag}");
        let kv_normed_dev = operators.cuda_rms_norm_device_cached(
            &kv_norm_name,
            &kv_dev,
            &self.payload.key_value_norm,
            cfg.norm_eps,
        )?;
        let mut kv = kv_normed_dev;
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut kv,
            position as u32,
            1,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        // Append to device KV cache.
        operators.cuda_ensure_kv_cache(self.layer, cfg.window_size, cfg.head_dim)?;
        operators.cuda_kv_append_device(
            self.layer,
            &kv,
            position,
            cfg.head_dim,
            cfg.window_size,
        )?;

        // Sparse attention on device (kv values fetched inside to avoid borrow conflict).
        let sink = &self.payload.attention_sink;
        let shape = ferrule_cuda::transformer::sparse_attention::CudaSparseAttentionShape {
            batch_size: 1,
            tokens_per_batch: 1,
            kv_len: cfg.window_size,
            heads: cfg.num_heads,
            head_dim: cfg.head_dim,
            topk: cfg.window_size,
            softmax_scale: (cfg.head_dim as f32).powf(-0.5),
        };
        let mut context = operators.cuda_sparse_attention_topk_from_device(
            &query,
            self.layer,
            position,
            cfg.window_size,
            sink,
            shape,
        )?;

        // Inverse rotary on context (device).
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut context,
            position as u32,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            true,
        )?;

        // Grouped output_a on device.
        let latent = operators.grouped_output_a_from_device(
            &self.payload.output_a,
            &context,
            cfg,
            self.layer,
        )?;

        // Output_b on device, then download final result.
        let mut latent = latent;
        let output =
            operators.cuda_linear_matvec_from_device(&self.payload.output_b, &mut latent)?;
        operators.cuda_download_f32(&output)
    }

    /// CUDA fully device-resident no-compress attention path.
    ///
    /// Like `decode_step_no_compress_device` but accepts and returns
    /// `CudaF32Buffer`, eliminating the boundary D2H+H2D round-trip.
    #[cfg(feature = "cuda")]
    fn decode_step_no_compress_from_device(
        &self,
        _cache: &mut DeepSeekV4WindowKvCache,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        operators: &mut DeepSeekV4OperatorContext,
        cfg: DeepSeekV4AttentionConfig,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let layer_tag = format!("attn_L{}", self.layer);
        let rope_name = format!("rope_{layer_tag}");
        operators.record_attention_call(self.layer, 1);

        operators.cuda_ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            4096,
        )?;
        let mut hidden_device = operators.cuda_clone_f32(hidden_dev)?;
        let mut kv_hidden_device = operators.cuda_clone_f32(hidden_dev)?;

        // Query: query_a → rms_norm → query_b → head_norm (all on device)
        let stage_start = Instant::now();
        let q_latent_dev =
            operators.cuda_linear_matvec_from_device(&self.payload.query_a, &mut hidden_device)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qa,
            stage_start,
        )?;
        let q_norm_name = format!("q_norm_{layer_tag}");
        let stage_start = Instant::now();
        let q_normed_dev = operators.cuda_rms_norm_device_cached(
            &q_norm_name,
            &q_latent_dev,
            &self.payload.query_norm,
            cfg.norm_eps,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QNorm,
            stage_start,
        )?;
        let mut q_normed_dev = q_normed_dev;
        let stage_start = Instant::now();
        let mut query =
            operators.cuda_linear_matvec_from_device(&self.payload.query_b, &mut q_normed_dev)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qb,
            stage_start,
        )?;
        let stage_start = Instant::now();
        query = operators.cuda_rms_norm_heads_from_device(
            &query,
            cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QHeadNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut query,
            position as u32,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QRope,
            stage_start,
        )?;

        // KV: key_value → rms_norm → rotary (all on device), then append to device KV cache.
        let stage_start = Instant::now();
        let kv_dev = operators
            .cuda_linear_matvec_from_device(&self.payload.key_value, &mut kv_hidden_device)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvProj,
            stage_start,
        )?;
        let kv_norm_name = format!("kv_norm_{layer_tag}");
        let stage_start = Instant::now();
        let kv_normed_dev = operators.cuda_rms_norm_device_cached(
            &kv_norm_name,
            &kv_dev,
            &self.payload.key_value_norm,
            cfg.norm_eps,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvNorm,
            stage_start,
        )?;
        let mut kv = kv_normed_dev;
        let stage_start = Instant::now();
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut kv,
            position as u32,
            1,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvRopeQuant,
            stage_start,
        )?;
        // Append to device KV cache.
        let stage_start = Instant::now();
        operators.cuda_ensure_kv_cache(self.layer, cfg.window_size, cfg.head_dim)?;
        operators.cuda_kv_append_device(
            self.layer,
            &kv,
            position,
            cfg.head_dim,
            cfg.window_size,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvCacheAppend,
            stage_start,
        )?;

        // Sparse attention on device.
        let sink = &self.payload.attention_sink;
        let shape = ferrule_cuda::transformer::sparse_attention::CudaSparseAttentionShape {
            batch_size: 1,
            tokens_per_batch: 1,
            kv_len: cfg.window_size,
            heads: cfg.num_heads,
            head_dim: cfg.head_dim,
            topk: cfg.window_size,
            softmax_scale: (cfg.head_dim as f32).powf(-0.5),
        };
        let stage_start = Instant::now();
        let mut context = operators.cuda_sparse_attention_topk_from_device(
            &query,
            self.layer,
            position,
            cfg.window_size,
            sink,
            shape,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::SparseAttention,
            stage_start,
        )?;

        // Inverse rotary on context (device).
        let stage_start = Instant::now();
        operators.cuda_rope_tail_from_device(
            &rope_name,
            &mut context,
            position as u32,
            cfg.num_heads as u32,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            true,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::ContextRope,
            stage_start,
        )?;

        // Grouped output_a on device.
        let stage_start = Instant::now();
        let latent = operators.grouped_output_a_from_device(
            &self.payload.output_a,
            &context,
            cfg,
            self.layer,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::OutputA,
            stage_start,
        )?;

        // Output_b on device, return device buffer (no D2H).
        let mut latent = latent;
        let stage_start = Instant::now();
        let output =
            operators.cuda_linear_matvec_from_device(&self.payload.output_b, &mut latent)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::OutputB,
            stage_start,
        )?;
        Ok(output)
    }
}

fn record_attention_stage(
    operators: &mut DeepSeekV4OperatorContext,
    layer: usize,
    stage: DeepSeekV4AttentionProfileStage,
    start: Instant,
) -> Result<()> {
    let elapsed_us = operators.finish_profile_stage(start)?;
    operators.record_attention_stage(layer, stage, elapsed_us);
    Ok(())
}

#[cfg(feature = "cuda")]
fn dsv4_fused_indexer_topk_enabled() -> bool {
    dsv4_env_flag("FERRULE_DSV4_FUSED_INDEXER_TOPK").unwrap_or(true)
}

#[cfg(feature = "cuda")]
fn dsv4_fused_indexer_topk_supported(cfg: DeepSeekV4AttentionConfig, rope_dim: usize) -> bool {
    dsv4_fused_indexer_prefill_topk_enabled()
        && dsv4_fused_indexer_topk_shape_supported(cfg, rope_dim)
}

#[cfg(feature = "cuda")]
fn dsv4_fused_indexer_prefill_topk_enabled() -> bool {
    dsv4_fused_indexer_topk_enabled()
        && dsv4_env_flag("FERRULE_DSV4_FUSED_INDEXER_TOPK_PREFILL").unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn dsv4_fused_indexer_decode_topk_supported(
    cfg: DeepSeekV4AttentionConfig,
    rope_dim: usize,
) -> bool {
    dsv4_fused_indexer_decode_topk_enabled()
        && dsv4_fused_indexer_topk_shape_supported(cfg, rope_dim)
}

#[cfg(feature = "cuda")]
fn dsv4_fused_indexer_decode_topk_enabled() -> bool {
    dsv4_fused_indexer_topk_enabled()
        && dsv4_env_flag("FERRULE_DSV4_FUSED_INDEXER_TOPK_DECODE").unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn dsv4_fused_indexer_topk_shape_supported(
    cfg: DeepSeekV4AttentionConfig,
    rope_dim: usize,
) -> bool {
    cfg.index_head_dim <= 256
        && cfg.index_head_dim.is_power_of_two()
        && cfg.index_head_dim.is_multiple_of(32)
        && rope_dim <= cfg.index_head_dim
        && rope_dim.is_multiple_of(2)
}

#[cfg(feature = "cuda")]
fn dsv4_env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|value| {
        let value = value.trim().to_ascii_lowercase();
        !(value.is_empty() || value == "0" || value == "false" || value == "off")
    })
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

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn compressed_len(&self) -> usize {
        self.compressed.len() / self.window.head_dim
    }

    pub fn indexer_compressed_len(&self, index_head_dim: usize) -> usize {
        self.indexer_compressed
            .len()
            .checked_div(index_head_dim)
            .unwrap_or(0)
    }

    pub fn reset_sequence(&mut self) {
        self.window.clear();
        self.compressed.clear();
        self.indexer_compressed.clear();
        if let Some(state) = &mut self.main_compressor {
            state.reset();
        }
        if let Some(state) = &mut self.indexer_compressor {
            state.reset();
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
pub struct DeepSeekV4CompressorState {
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

    fn reset(&mut self) {
        self.kv_state.fill(0.0);
        self.score_state.fill(f32::NEG_INFINITY);
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn update_prefill_shadow_state(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        tokens: usize,
        kv_rows: &[f32],
        score_rows: &[f32],
    ) -> Result<usize> {
        if kv_rows.len() != tokens * self.out_dim || score_rows.len() != tokens * self.out_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 compressor prefill rows shadow length mismatch: kv={} score={} expected {}x{}",
                kv_rows.len(),
                score_rows.len(),
                tokens,
                self.out_dim
            )));
        }
        self.kv_state.fill(0.0);
        self.score_state.fill(f32::NEG_INFINITY);

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
        Ok(groups)
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
        if hidden.is_empty()
            || !hidden
                .len()
                .is_multiple_of(payload.wkv.format.in_features())
        {
            return Err(Error::Model(format!(
                "DeepSeek-V4 compressor prefill hidden length mismatch: hidden={} in_features={}",
                hidden.len(),
                payload.wkv.format.in_features()
            )));
        }
        self.kv_state.fill(0.0);
        self.score_state.fill(f32::NEG_INFINITY);

        let tokens = hidden.len() / payload.wkv.format.in_features();
        let kv_rows = operators.linear_rows(&payload.wkv, hidden, tokens)?;
        let score_rows = operators.linear_rows(&payload.wgate, hidden, tokens)?;
        self.prefill_start_projected(
            payload, tokens, kv_rows, score_rows, rope_dim, rope, operators,
        )
    }

    #[cfg(feature = "cuda")]
    fn prefill_start_device_output(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        rope_name: &str,
        norm_name: &str,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<(ferrule_cuda::context::CudaF32Buffer, Vec<f32>)> {
        if payload.compress_ratio != self.ratio
            || payload.head_dim != self.head_dim
            || payload.overlap != self.overlap
        {
            return Err(Error::Model(
                "DeepSeek-V4 compressor payload/state shape mismatch".into(),
            ));
        }
        if hidden_dev.len() == 0
            || !hidden_dev
                .len()
                .is_multiple_of(payload.wkv.format.in_features())
        {
            return Err(Error::Model(format!(
                "DeepSeek-V4 compressor device prefill hidden length mismatch: hidden={} in_features={}",
                hidden_dev.len(),
                payload.wkv.format.in_features()
            )));
        }

        let tokens = hidden_dev.len() / payload.wkv.format.in_features();
        let kv_rows_dev = operators.linear_rows_from_device(&payload.wkv, hidden_dev, tokens)?;
        let score_rows_dev =
            operators.linear_rows_from_device(&payload.wgate, hidden_dev, tokens)?;

        // The decode append path still keeps a compact CPU shadow state. We only
        // download projected rows for that shadow; compression itself stays on GPU.
        let kv_rows = operators.cuda_download_f32(&kv_rows_dev)?;
        let score_rows = operators.cuda_download_f32(&score_rows_dev)?;
        let groups = self.update_prefill_shadow_state(payload, tokens, &kv_rows, &score_rows)?;
        if groups == 0 {
            let empty = operators.cuda_upload_f32(&[])?;
            return Ok((empty, Vec::new()));
        }

        let mut compressed = operators.cuda_compressor_prefill_softmax_from_device(
            &kv_rows_dev,
            &score_rows_dev,
            &payload.ape,
            groups,
            self.ratio,
            self.head_dim,
            self.out_dim,
            self.overlap,
        )?;
        compressed = operators.cuda_rms_norm_rows_device_cached(
            norm_name,
            &compressed,
            groups,
            &payload.norm,
            1e-6,
        )?;
        let effective_rope_dim = rope_dim.min(self.head_dim);
        operators.cuda_ensure_rope_tables_with_params(rope_name, effective_rope_dim, rope, 4096)?;
        operators.cuda_rope_tail_rows_strided_from_device(
            rope_name,
            &mut compressed,
            0,
            self.ratio as u32,
            groups as u32,
            1,
            self.head_dim as u32,
            effective_rope_dim as u32,
            false,
        )?;
        if payload.rotate_for_indexer {
            operators.cuda_indexer_qat_quantize_buffer_in_place(&mut compressed, self.head_dim)?;
        } else {
            operators.cuda_attention_kv_qat_quantize_buffer_in_place(
                &mut compressed,
                self.head_dim,
                effective_rope_dim,
            )?;
        }
        let host_shadow = operators.cuda_download_f32(&compressed)?;
        Ok((compressed, host_shadow))
    }

    fn prefill_start_projected(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        tokens: usize,
        kv_rows: Vec<f32>,
        score_rows: Vec<f32>,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        if kv_rows.len() != tokens * self.out_dim || score_rows.len() != tokens * self.out_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 compressor prefill rows matvec length mismatch: kv={} score={} expected {}x{}",
                kv_rows.len(),
                score_rows.len(),
                tokens,
                self.out_dim
            )));
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
            quantize_compressed_kv_for_qat_in_place(
                &mut out,
                self.head_dim,
                rope_dim.min(self.head_dim),
                payload.rotate_for_indexer,
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
        self.validate_append_payload(payload)?;
        let kv = operators.linear_matvec(&payload.wkv, hidden)?;
        let score = operators.linear_matvec(&payload.wgate, hidden)?;
        self.append_projected_step(payload, kv, score, position, rope_dim, rope, operators)
    }

    #[cfg(feature = "cuda")]
    fn append_step_from_device_if_unquantized(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        hidden_device: &mut ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Option<Vec<f32>>> {
        self.validate_append_payload(payload)?;
        if payload.wkv.execution.activation_quantization.is_some()
            || payload.wgate.execution.activation_quantization.is_some()
        {
            return Err(Error::Model(
                "DeepSeek-V4 compressor device append requires unquantized activation inputs"
                    .into(),
            ));
        }
        let kv_dev = operators.cuda_linear_matvec_from_device(&payload.wkv, hidden_device)?;
        let score_dev = operators.cuda_linear_matvec_from_device(&payload.wgate, hidden_device)?;
        let kv = operators.cuda_download_f32(&kv_dev)?;
        let score = operators.cuda_download_f32(&score_dev)?;
        self.append_projected_step(payload, kv, score, position, rope_dim, rope, operators)
    }

    #[cfg(feature = "cuda")]
    fn append_step_from_device(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Option<Vec<f32>>> {
        self.validate_append_payload(payload)?;
        let mut kv_input = operators.cuda_clone_f32(hidden_dev)?;
        let kv_dev = operators.cuda_linear_matvec_from_device(&payload.wkv, &mut kv_input)?;
        let mut score_input = operators.cuda_clone_f32(hidden_dev)?;
        let score_dev =
            operators.cuda_linear_matvec_from_device(&payload.wgate, &mut score_input)?;
        let kv = operators.cuda_download_f32(&kv_dev)?;
        let score = operators.cuda_download_f32(&score_dev)?;
        self.append_projected_step(payload, kv, score, position, rope_dim, rope, operators)
    }

    fn validate_append_payload(&self, payload: &DeepSeekV4CompressorPayload) -> Result<()> {
        if payload.compress_ratio != self.ratio
            || payload.head_dim != self.head_dim
            || payload.overlap != self.overlap
        {
            return Err(Error::Model(
                "DeepSeek-V4 compressor payload/state shape mismatch".into(),
            ));
        }
        Ok(())
    }

    fn append_projected_step(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        kv: Vec<f32>,
        mut score: Vec<f32>,
        position: usize,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Option<Vec<f32>>> {
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

        if !(position + 1).is_multiple_of(self.ratio) {
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
        quantize_compressed_kv_for_qat_in_place(
            &mut out,
            self.head_dim,
            rope_dim.min(self.head_dim),
            payload.rotate_for_indexer,
        )?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4WindowKvCache {
    window_size: usize,
    pub(crate) head_dim: usize,
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

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.values.fill(0.0);
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

    pub(crate) fn values_full(&self) -> &[f32] {
        &self.values
    }
}
