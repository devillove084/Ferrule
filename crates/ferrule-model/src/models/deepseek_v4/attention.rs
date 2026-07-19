//! DeepSeek-V4 attention: MLA, compressor, indexer, window KV cache.

use std::time::Instant;

use crate::artifact::binding::MlaAttentionArtifactPayload;
use crate::artifact::linear::ArtifactLinearPayload;
use crate::artifact::tensor::{ArtifactTensorReader, ArtifactTensorSlice};
#[cfg(feature = "cuda")]
use crate::attention_backend::SparseAttentionSpec;

#[cfg(feature = "cuda")]
use ferrule_common::execution::ForwardPhase;

use ferrule_common::{Error, Result};

use super::config::{DeepSeekV4AttentionConfig, DeepSeekV4RopeParams};
#[cfg(all(feature = "cuda", feature = "cutlass"))]
use super::cuda_cache::DeepSeekV4DsparkAttentionBuffers;
#[cfg(feature = "cuda")]
use super::cuda_cache::{
    DeepSeekV4CudaCompressor, DeepSeekV4CudaLayerNorm, DeepSeekV4CudaLinear,
    DeepSeekV4CudaSequenceKvState,
};
use super::helpers::{
    apply_rotary_tail, apply_rotary_tail_scaled, bind_aux_linear, check_len, check_linear,
    compress_rows_softmax, compress_topk_indices_prefill, concat_topk_rows, decode_tensor_f32,
    decode_vector_f32, indexer_topk_indices, indexer_topk_indices_prefill,
    quantize_attention_kv_for_qat_in_place, quantize_compressed_kv_for_qat_in_place,
    read_aux_tensor, read_aux_tensor_f32, two_dim_shape_from_payload, window_topk_indices_prefill,
};
use super::operators::{DeepSeekV4AttentionProfileStage, DeepSeekV4OperatorContext};
#[cfg(feature = "cuda")]
use super::sequence::DeepSeekV4PagedKvBinding;
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
struct DeepSeekV4CompressorDecodeArena {
    kv: ferrule_cuda::context::CudaF32Buffer,
    score: ferrule_cuda::context::CudaF32Buffer,
    compressed: ferrule_cuda::context::CudaF32Buffer,
    normalized: ferrule_cuda::context::CudaF32Buffer,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4CompressorDecodeArena {
    fn new(
        payload: &DeepSeekV4CompressorPayload,
        rows: usize,
        independent_sequences: bool,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Self> {
        let ops = &operators.cuda_mut()?.ops;
        let boundary_rows = if independent_sequences {
            rows
        } else {
            rows / payload.compress_ratio
        };
        Ok(Self {
            kv: ops.zero_f32_buffer(rows * payload.wkv.format.out_features())?,
            score: ops.zero_f32_buffer(rows * payload.wgate.format.out_features())?,
            compressed: ops.zero_f32_buffer(boundary_rows * payload.head_dim)?,
            normalized: ops.zero_f32_buffer(boundary_rows * payload.head_dim)?,
        })
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4AttentionRowsTransitionArena {
    input: ferrule_cuda::context::CudaF32Buffer,
    main_compressor: Option<DeepSeekV4CompressorDecodeArena>,
    indexer_compressor: Option<DeepSeekV4CompressorDecodeArena>,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4AttentionRowsTransitionArena {
    pub(crate) fn new(
        attention: &DeepSeekV4Attention,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Self> {
        let cfg = attention.config;
        let main_compressor = attention
            .compressed
            .as_ref()
            .map(|payload| {
                DeepSeekV4CompressorDecodeArena::new(&payload.compressor, 1, true, operators)
            })
            .transpose()?;
        let indexer_compressor = attention
            .compressed
            .as_ref()
            .and_then(|payload| payload.indexer.as_ref())
            .map(|indexer| {
                DeepSeekV4CompressorDecodeArena::new(&indexer.compressor, 1, true, operators)
            })
            .transpose()?;
        Ok(Self {
            input: operators.cuda_mut()?.ops.zero_f32_buffer(cfg.hidden_size)?,
            main_compressor,
            indexer_compressor,
        })
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4AttentionDecodeArena {
    hidden_a: ferrule_cuda::context::CudaF32Buffer,
    q_latent: ferrule_cuda::context::CudaF32Buffer,
    q_norm: ferrule_cuda::context::CudaF32Buffer,
    q_indexer: ferrule_cuda::context::CudaF32Buffer,
    query_raw: ferrule_cuda::context::CudaF32Buffer,
    query: ferrule_cuda::context::CudaF32Buffer,
    kv_raw: ferrule_cuda::context::CudaF32Buffer,
    kv: ferrule_cuda::context::CudaF32Buffer,
    index_query: ferrule_cuda::context::CudaF32Buffer,
    index_weights: ferrule_cuda::context::CudaF32Buffer,
    positions: ferrule_cuda::context::CudaI32HostMirror,
    window_lens: ferrule_cuda::context::CudaI32HostMirror,
    compressed_lens: ferrule_cuda::context::CudaI32HostMirror,
    visible_lens: ferrule_cuda::context::CudaI32HostMirror,
    main_positions: ferrule_cuda::context::CudaI32HostMirror,
    main_mask: ferrule_cuda::context::CudaI32HostMirror,
    indexer_positions: ferrule_cuda::context::CudaI32HostMirror,
    indexer_mask: ferrule_cuda::context::CudaI32HostMirror,
    topk: ferrule_cuda::context::CudaI32Buffer,
    topk_selectors: ferrule_cuda::context::CudaI32Buffer,
    context: ferrule_cuda::context::CudaF32Buffer,
    latent: ferrule_cuda::context::CudaF32Buffer,
    pub(crate) output: ferrule_cuda::context::CudaF32Buffer,
    compact_values: ferrule_cuda::context::CudaF32Buffer,
    pub(crate) linear_workspace: ferrule_cuda::context::CudaArtifactLinearWorkspace,
    main_compressor: Option<DeepSeekV4CompressorDecodeArena>,
    indexer_compressor: Option<DeepSeekV4CompressorDecodeArena>,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4AttentionDecodeArena {
    pub(crate) fn new(
        attention: &DeepSeekV4Attention,
        rows: usize,
        independent_sequences: bool,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Self> {
        let cfg = attention.config;
        let main_compressor = attention
            .compressed
            .as_ref()
            .map(|payload| {
                DeepSeekV4CompressorDecodeArena::new(
                    &payload.compressor,
                    rows,
                    independent_sequences,
                    operators,
                )
            })
            .transpose()?;
        let indexer_compressor = attention
            .compressed
            .as_ref()
            .and_then(|payload| payload.indexer.as_ref())
            .map(|indexer| {
                DeepSeekV4CompressorDecodeArena::new(
                    &indexer.compressor,
                    rows,
                    independent_sequences,
                    operators,
                )
            })
            .transpose()?;
        let ops = &operators.cuda_mut()?.ops;
        let compressed_rows = if cfg.compress_ratio == 0 {
            0
        } else if independent_sequences {
            rows
        } else {
            rows / cfg.compress_ratio
        };
        let max_linear_width = cfg
            .hidden_size
            .max(cfg.q_lora_rank)
            .max(cfg.q_full_dim())
            .max(cfg.head_dim)
            .max(cfg.output_latent_dim())
            .max(cfg.index_n_heads * cfg.index_head_dim);
        let zero_control_rows = vec![0i32; rows];
        Ok(Self {
            hidden_a: ops.zero_f32_buffer(rows * cfg.hidden_size)?,
            q_latent: ops.zero_f32_buffer(rows * cfg.q_lora_rank)?,
            q_norm: ops.zero_f32_buffer(rows * cfg.q_lora_rank)?,
            q_indexer: ops.zero_f32_buffer(rows * cfg.q_lora_rank)?,
            query_raw: ops.zero_f32_buffer(rows * cfg.q_full_dim())?,
            query: ops.zero_f32_buffer(rows * cfg.q_full_dim())?,
            kv_raw: ops.zero_f32_buffer(rows * cfg.head_dim)?,
            kv: ops.zero_f32_buffer(rows * cfg.head_dim)?,
            index_query: ops.zero_f32_buffer(rows * cfg.index_n_heads * cfg.index_head_dim)?,
            index_weights: ops.zero_f32_buffer(rows * cfg.index_n_heads)?,
            positions: ops.i32_host_mirror(&zero_control_rows)?,
            window_lens: ops.i32_host_mirror(&zero_control_rows)?,
            compressed_lens: ops.i32_host_mirror(&zero_control_rows)?,
            visible_lens: ops.i32_host_mirror(&zero_control_rows)?,
            main_positions: ops.i32_host_mirror(&zero_control_rows)?,
            main_mask: ops.i32_host_mirror(&zero_control_rows)?,
            indexer_positions: ops.i32_host_mirror(&zero_control_rows)?,
            indexer_mask: ops.i32_host_mirror(&zero_control_rows)?,
            topk: ops.zero_i32_buffer(rows * (cfg.window_size + cfg.index_topk))?,
            topk_selectors: ops.zero_i32_buffer(rows * (cfg.window_size + cfg.index_topk))?,
            context: ops.zero_f32_buffer(rows * cfg.q_full_dim())?,
            latent: ops.zero_f32_buffer(rows * cfg.output_latent_dim())?,
            output: ops.zero_f32_buffer(rows * cfg.hidden_size)?,
            compact_values: ops.zero_f32_buffer((rows + compressed_rows) * cfg.head_dim)?,
            linear_workspace: ops.artifact_linear_workspace(rows, max_linear_width)?,
            main_compressor,
            indexer_compressor,
        })
    }
}

#[cfg(feature = "cuda")]
fn decode_metadata_i32(values: &[usize], label: &str) -> Result<Vec<i32>> {
    values
        .iter()
        .map(|&value| {
            i32::try_from(value)
                .map_err(|_| Error::Model(format!("packed decode {label} exceeds i32 ABI")))
        })
        .collect()
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
                self.layer,
                cache.window.window_size,
                cache.window.head_dim,
                cfg.window_size,
                cfg.head_dim
            )));
        }

        let stage_start = operators.profile_start();
        let q_latents = operators.linear_rows(&self.payload.query_a, hidden, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qa,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
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
        let stage_start = operators.profile_start();
        let mut queries = operators.linear_rows(&self.payload.query_b, &q_latents, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qb,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
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
        let stage_start = operators.profile_start();
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

        let stage_start = operators.profile_start();
        let kv_rows = operators.linear_rows(&self.payload.key_value, hidden, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvProj,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
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
        let stage_start = operators.profile_start();
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
        let stage_start = operators.profile_start();
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

        let stage_start = operators.profile_start();
        let topk_cols = tokens.min(cfg.window_size);
        let topk = window_topk_indices_prefill(cfg.window_size, tokens);
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::TopkBuild,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
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
                self.layer,
                cache.window.window_size,
                cache.window.head_dim,
                cfg.window_size,
                cfg.head_dim
            )));
        }
        let compressed = self.compressed.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {} has compress_ratio {} but no typed compressed attention payload is bound",
                self.layer, cfg.compress_ratio
            ))
        })?;
        let rope = cfg.rope_params();

        let stage_start = operators.profile_start();
        let q_latents = operators.linear_rows(&self.payload.query_a, hidden, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qa,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
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
        let stage_start = operators.profile_start();
        let mut queries = operators.linear_rows(&self.payload.query_b, &q_latents, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qb,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
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
        let stage_start = operators.profile_start();
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

        let stage_start = operators.profile_start();
        let kv_rows = operators.linear_rows(&self.payload.key_value, hidden, tokens)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvProj,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
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
        let stage_start = operators.profile_start();
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
        let stage_start = operators.profile_start();
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
            let stage_start = operators.profile_start();
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

        let stage_start = operators.profile_start();
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

        let stage_start = operators.profile_start();
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
        let stage_start = operators.profile_start();
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
        let mut out = Vec::with_capacity(tokens * cfg.hidden_size);
        for token in 0..tokens {
            let position = start_pos + token;
            let row_start = token * cfg.q_full_dim();
            let row = &mut context[row_start..row_start + cfg.q_full_dim()];
            let stage_start = operators.profile_start();
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
            let stage_start = operators.profile_start();
            let latent =
                operators.grouped_output_a(&self.payload.output_a, row, cfg, self.layer)?;
            record_attention_stage(
                operators,
                self.layer,
                DeepSeekV4AttentionProfileStage::OutputA,
                stage_start,
            )?;
            let stage_start = operators.profile_start();
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

    #[cfg(feature = "cuda")]
    pub(crate) fn prefill_start_from_device_into(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        hidden_fp8: &ferrule_cuda::context::CudaPreparedFp8Activation<'_>,
        start_pos: usize,
        arena: &mut DeepSeekV4AttentionDecodeArena,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<()> {
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
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} continuation prefill has no SM121 semantic attention plan",
                self.layer
            )));
        }
        if cfg.compress_ratio == 0 {
            self.prefill_start_no_compress_from_device_into(
                cache, hidden_dev, hidden_fp8, start_pos, arena, operators,
            )
        } else {
            self.prefill_start_compressed_from_device_into(
                cache, hidden_dev, hidden_fp8, start_pos, arena, operators,
            )
        }
    }

    #[cfg(feature = "cuda")]
    fn prefill_start_no_compress_from_device_into(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        hidden_fp8: &ferrule_cuda::context::CudaPreparedFp8Activation<'_>,
        start_pos: usize,
        arena: &mut DeepSeekV4AttentionDecodeArena,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<()> {
        let cfg = self.config;
        let tokens = hidden_dev.len() / cfg.hidden_size;
        operators.record_attention_call(self.layer, tokens);
        if cache.window.head_dim != cfg.head_dim || cache.window.window_size != cfg.window_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} KV cache shape mismatch: cache window={} head_dim={}, expected window={} head_dim={}",
                self.layer,
                cache.window.window_size,
                cache.window.head_dim,
                cfg.window_size,
                cfg.head_dim
            )));
        }
        let layer_tag = format!("attn_L{}", self.layer);

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.query_a_kv_from_prepared_fp8_into(
            self.layer,
            hidden_fp8,
            &mut arena.q_latent,
            &mut arena.kv_raw,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qa,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_layer_rows_device_into(
            self.layer,
            DeepSeekV4CudaLayerNorm::Query,
            &arena.q_latent,
            tokens,
            cfg.norm_eps,
            &mut arena.q_norm,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QNorm,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.linear_rows_from_device_into(
            self.layer,
            DeepSeekV4CudaLinear::QueryB,
            &arena.q_norm,
            tokens,
            &mut arena.query_raw,
            &mut arena.linear_workspace,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qb,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_heads_from_device_into(
            &arena.query_raw,
            tokens * cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
            &mut arena.query,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QHeadNorm,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        let rope_name = format!("rope_{layer_tag}");
        let rope_positions = required_rope_positions(start_pos, 1, tokens)?;
        operators.cuda_mut()?.ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            rope_positions,
        )?;
        operators.cuda_mut()?.rope_tail_rows_from_device(
            &rope_name,
            &mut arena.query,
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

        let stage_start = operators.profile_start();
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvProj,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_layer_rows_device_into(
            self.layer,
            DeepSeekV4CudaLayerNorm::KeyValue,
            &arena.kv_raw,
            tokens,
            cfg.norm_eps,
            &mut arena.kv,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvNorm,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rope_tail_rows_from_device(
            &rope_name,
            &mut arena.kv,
            start_pos as u32,
            tokens as u32,
            1,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        operators
            .cuda_mut()?
            .ops
            .fp8_attention_kv_qat_quantize_buffer_in_place(
                &mut arena.kv,
                cfg.head_dim,
                cfg.rope_head_dim,
            )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvRopeQuant,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        let zero_kv = vec![0.0f32; cfg.head_dim];
        for token in 0..tokens {
            cache.window.append(start_pos + token, &zero_kv)?;
        }
        operators.cuda_mut()?.paged_write_rows(
            0,
            self.layer,
            &arena.kv,
            start_pos,
            tokens,
            cfg.head_dim,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvCacheAppend,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        let topk_cols = tokens.min(cfg.window_size);
        let topk = window_topk_indices_prefill(cfg.window_size, tokens);
        operators
            .cuda_mut()?
            .write_topk_indices(&topk, &mut arena.topk)?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::TopkBuild,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators
            .cuda_mut()?
            .sparse_attention_with_device_query_values_topk_into(
                &arena.query,
                &arena.kv,
                &arena.topk,
                tokens,
                tokens,
                cfg.sparse_spec_with_topk(topk_cols),
                self.layer,
                &mut arena.context,
            )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::SparseAttention,
            stage_start,
        )?;
        self.project_context_rows_device_to_device_into(arena, start_pos, tokens, operators)
    }

    #[cfg(feature = "cuda")]
    fn prefill_start_compressed_from_device_into(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        hidden_fp8: &ferrule_cuda::context::CudaPreparedFp8Activation<'_>,
        start_pos: usize,
        arena: &mut DeepSeekV4AttentionDecodeArena,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<()> {
        let cfg = self.config;
        let tokens = hidden_dev.len() / cfg.hidden_size;
        operators.record_attention_call(self.layer, tokens);
        if cache.window.head_dim != cfg.head_dim || cache.window.window_size != cfg.window_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} KV cache shape mismatch: cache window={} head_dim={}, expected window={} head_dim={}",
                self.layer,
                cache.window.window_size,
                cache.window.head_dim,
                cfg.window_size,
                cfg.head_dim
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

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.query_a_kv_from_prepared_fp8_into(
            self.layer,
            hidden_fp8,
            &mut arena.q_latent,
            &mut arena.kv_raw,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qa,
            stage_start,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_q_latent",
            &arena.q_latent,
            cfg.q_lora_rank,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_layer_rows_device_into(
            self.layer,
            DeepSeekV4CudaLayerNorm::Query,
            &arena.q_latent,
            tokens,
            cfg.norm_eps,
            &mut arena.q_norm,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QNorm,
            stage_start,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_q_norm",
            &arena.q_norm,
            cfg.q_lora_rank,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.linear_rows_from_device_into(
            self.layer,
            DeepSeekV4CudaLinear::QueryB,
            &arena.q_norm,
            tokens,
            &mut arena.query_raw,
            &mut arena.linear_workspace,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qb,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_heads_from_device_into(
            &arena.query_raw,
            tokens * cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
            &mut arena.query,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QHeadNorm,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        let rope_name = format!("rope_{layer_tag}");
        let rope_positions = required_rope_positions(start_pos, 1, tokens)?;
        operators.cuda_mut()?.ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            rope_positions,
        )?;
        operators.cuda_mut()?.rope_tail_rows_from_device(
            &rope_name,
            &mut arena.query,
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
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_query",
            &arena.query,
            cfg.q_full_dim(),
        )?;

        let stage_start = operators.profile_start();
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvProj,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_layer_rows_device_into(
            self.layer,
            DeepSeekV4CudaLayerNorm::KeyValue,
            &arena.kv_raw,
            tokens,
            cfg.norm_eps,
            &mut arena.kv,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvNorm,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rope_tail_rows_from_device(
            &rope_name,
            &mut arena.kv,
            start_pos as u32,
            tokens as u32,
            1,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        operators
            .cuda_mut()?
            .ops
            .fp8_attention_kv_qat_quantize_buffer_in_place(
                &mut arena.kv,
                cfg.head_dim,
                cfg.rope_head_dim,
            )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvRopeQuant,
            stage_start,
        )?;
        operators.capture_parity_checkpoint_rows(self.layer, "attn_kv", &arena.kv, cfg.head_dim)?;
        let stage_start = operators.profile_start();
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

        if let Some(indexer) = compressed.indexer.as_ref() {
            let indexer_compressed_start = cache.indexer_compressed_len(cfg.index_head_dim);
            let stage_start = operators.profile_start();
            let indexer_groups = {
                let (compressor_state, window) = (&mut cache.indexer_compressor, &mut cache.window);
                let state = compressor_state.as_mut().ok_or_else(|| {
                    Error::Model(format!(
                        "DeepSeek-V4 layer {} missing indexer compressor state",
                        self.layer
                    ))
                })?;
                let compressor_arena = arena.indexer_compressor.as_mut().ok_or_else(|| {
                    Error::Model(format!(
                        "DeepSeek-V4 layer {} missing indexer compressor arena",
                        self.layer
                    ))
                })?;
                let cuda = window.cuda_state_mut();
                state.prefill_start_device_output_into(
                    &indexer.compressor,
                    self.layer,
                    DeepSeekV4CudaCompressor::Indexer,
                    hidden_dev,
                    cfg.rope_head_dim,
                    rope,
                    &format!("rope_indexer_compress_L{}", self.layer),
                    compressor_arena,
                    &mut arena.linear_workspace,
                    &mut cuda.indexer_compressor_recurrent,
                    &mut cuda.indexer_compressor_needs_reset,
                    operators,
                )?
            };
            record_attention_stage(
                operators,
                self.layer,
                DeepSeekV4AttentionProfileStage::IndexerCompress,
                stage_start,
            )?;
            cache.indexer_compressed.resize(
                cache.indexer_compressed.len() + indexer_groups * cfg.index_head_dim,
                0.0,
            );
            let indexer_rows = arena
                .indexer_compressor
                .as_ref()
                .expect("indexer compressor arena exists")
                .normalized
                .len()
                / cfg.index_head_dim;
            operators.cuda_mut()?.paged_write_rows(
                2,
                self.layer,
                &arena
                    .indexer_compressor
                    .as_ref()
                    .expect("indexer compressor arena exists")
                    .normalized,
                indexer_compressed_start,
                indexer_rows,
                cfg.index_head_dim,
            )?;
        }

        let stage_start = operators.profile_start();
        let compressed_start = cache.compressed_len();
        let main_groups = {
            let (compressor_state, window) = (&mut cache.main_compressor, &mut cache.window);
            let state = compressor_state.as_mut().ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 layer {} missing main compressor state",
                    self.layer
                ))
            })?;
            let compressor_arena = arena.main_compressor.as_mut().ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 layer {} missing main compressor arena",
                    self.layer
                ))
            })?;
            let cuda = window.cuda_state_mut();
            state.prefill_start_device_output_into(
                &compressed.compressor,
                self.layer,
                DeepSeekV4CudaCompressor::Main,
                hidden_dev,
                cfg.rope_head_dim,
                rope,
                &format!("rope_main_compress_L{}", self.layer),
                compressor_arena,
                &mut arena.linear_workspace,
                &mut cuda.main_compressor_recurrent,
                &mut cuda.main_compressor_needs_reset,
                operators,
            )?
        };
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::MainCompress,
            stage_start,
        )?;
        cache
            .compressed
            .resize(cache.compressed.len() + main_groups * cfg.head_dim, 0.0);

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.paged_write_rows(
            0,
            self.layer,
            &arena.kv,
            start_pos,
            tokens,
            cfg.head_dim,
        )?;
        let main_compressed = &arena
            .main_compressor
            .as_ref()
            .expect("main compressor arena exists")
            .normalized;
        operators.cuda_mut()?.paged_write_rows(
            1,
            self.layer,
            main_compressed,
            compressed_start,
            main_compressed.len() / cfg.head_dim,
            cfg.head_dim,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::CompressedKvUpload,
            stage_start,
        )?;

        let main_compressed_len = arena
            .main_compressor
            .as_ref()
            .expect("main compressor arena exists")
            .normalized
            .len()
            / cfg.head_dim;
        let stage_start = operators.profile_start();
        let window_cols = tokens.min(cfg.window_size);
        let compressed_offset = tokens;
        let topk_cols = if compressed.indexer.is_some() {
            let indexer_kv = &arena
                .indexer_compressor
                .as_ref()
                .expect("indexer compressor arena exists")
                .normalized;
            let compressed_len = indexer_kv.len() / cfg.index_head_dim;
            let cached_compressed_len = cache.indexer_compressed_len(cfg.index_head_dim);
            if compressed_len != cached_compressed_len {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 layer {} device indexer compressed length mismatch: device={compressed_len} cache={cached_compressed_len}",
                    self.layer
                )));
            }
            let indexer_cols = cfg.index_topk.min(compressed_len);
            if indexer_cols == 0 {
                let topk = window_topk_indices_prefill(cfg.window_size, tokens);
                operators
                    .cuda_mut()?
                    .write_topk_indices(&topk, &mut arena.topk)?;
            } else {
                operators.cuda_mut()?.linear_rows_from_device_into(
                    self.layer,
                    DeepSeekV4CudaLinear::IndexerQuery,
                    &arena.q_norm,
                    tokens,
                    &mut arena.index_query,
                    &mut arena.linear_workspace,
                )?;
                operators.cuda_mut()?.linear_rows_from_device_into(
                    self.layer,
                    DeepSeekV4CudaLinear::IndexerWeights,
                    hidden_dev,
                    tokens,
                    &mut arena.index_weights,
                    &mut arena.linear_workspace,
                )?;
                let index_rope_dim = cfg.rope_head_dim.min(cfg.index_head_dim);
                let index_rope_name = format!("rope_indexer_query_L{}", self.layer);
                operators.cuda_mut()?.ensure_rope_tables_with_params(
                    &index_rope_name,
                    index_rope_dim,
                    cfg.rope_params(),
                    rope_positions,
                )?;
                let weight_scale =
                    (cfg.index_head_dim as f32).powf(-0.5) * (cfg.index_n_heads as f32).powf(-0.5);
                if dsv4_fused_indexer_topk_supported(operators, cfg, index_rope_dim) {
                    operators
                        .cuda_mut()?
                        .prefill_topk_indices_fused_index_query_paged_indexer_into(
                            &arena.index_query,
                            &arena.index_weights,
                            self.layer,
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
                            &mut arena.topk,
                        )?;
                } else {
                    operators.cuda_mut()?.rope_tail_rows_from_device(
                        &index_rope_name,
                        &mut arena.index_query,
                        start_pos as u32,
                        tokens as u32,
                        cfg.index_n_heads as u32,
                        cfg.index_head_dim as u32,
                        index_rope_dim as u32,
                        false,
                    )?;
                    operators
                        .cuda_mut()?
                        .ops
                        .fp4_hadamard_qat_quantize_buffer_in_place(
                            &mut arena.index_query,
                            cfg.index_head_dim,
                        )?;
                    operators
                        .cuda_mut()?
                        .prefill_topk_indices_paged_indexer_into(
                            &arena.index_query,
                            &arena.index_weights,
                            self.layer,
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
                            &mut arena.topk,
                        )?;
                }
            }
            window_cols + indexer_cols
        } else {
            let mut topk = window_topk_indices_prefill(cfg.window_size, tokens);
            let (compressed_topk, compressed_cols) =
                compress_topk_indices_prefill(cfg.compress_ratio, tokens, compressed_offset);
            if compressed_cols > 0 {
                topk = concat_topk_rows(
                    &topk,
                    window_cols,
                    &compressed_topk,
                    compressed_cols,
                    tokens,
                )?;
            }
            operators
                .cuda_mut()?
                .write_topk_indices(&topk, &mut arena.topk)?;
            window_cols + main_compressed_len
        };
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::TopkBuild,
            stage_start,
        )?;

        operators
            .cuda_mut()?
            .concat_attention_values_device_buffers_into(
                &arena.kv,
                &arena
                    .main_compressor
                    .as_ref()
                    .expect("main compressor arena exists")
                    .normalized,
                tokens,
                cfg.head_dim,
                &mut arena.compact_values,
            )?;
        let stage_start = operators.profile_start();
        operators
            .cuda_mut()?
            .sparse_attention_with_device_query_values_topk_into(
                &arena.query,
                &arena.compact_values,
                &arena.topk,
                tokens,
                tokens + main_compressed_len,
                cfg.sparse_spec_with_topk(topk_cols),
                self.layer,
                &mut arena.context,
            )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::SparseAttention,
            stage_start,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_context",
            &arena.context,
            cfg.q_full_dim(),
        )?;
        self.project_context_rows_device_to_device_into(arena, start_pos, tokens, operators)
    }

    #[cfg(feature = "cuda")]
    fn project_context_rows_device_to_device_into(
        &self,
        arena: &mut DeepSeekV4AttentionDecodeArena,
        start_pos: usize,
        tokens: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<()> {
        let cfg = self.config;
        if arena.context.len() != tokens * cfg.q_full_dim() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} device attention context length mismatch: expected {}, got {}",
                self.layer,
                tokens * cfg.q_full_dim(),
                arena.context.len()
            )));
        }
        let stage_start = operators.profile_start();
        let rope_name = format!("rope_attn_L{}", self.layer);
        let rope_positions = required_rope_positions(start_pos, 1, tokens)?;
        operators.cuda_mut()?.ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            rope_positions,
        )?;
        operators.cuda_mut()?.rope_tail_rows_from_device(
            &rope_name,
            &mut arena.context,
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
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_context_rope",
            &arena.context,
            cfg.q_full_dim(),
        )?;

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.mla_output_rows_from_device_into(
            &arena.context,
            tokens,
            cfg,
            self.layer,
            &mut arena.latent,
            &mut arena.linear_workspace,
            &mut arena.output,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::OutputB,
            stage_start,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_output_a",
            &arena.latent,
            self.payload.output_b.format.in_features(),
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_output",
            &arena.output,
            cfg.hidden_size,
        )?;
        Ok(())
    }

    pub fn decode_step_reference(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden: &[f32],
        position: usize,
    ) -> Result<Vec<f32>> {
        let mut operators = DeepSeekV4OperatorContext::new_cpu()?;
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
                self.layer,
                cache.window.window_size,
                cache.window.head_dim,
                cfg.window_size,
                cfg.head_dim
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
            #[cfg(feature = "cuda")]
            operators.fail_compressor_transition_if_armed(true)?;
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
        #[cfg(feature = "cuda")]
        operators.fail_compressor_transition_if_armed(false)?;
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

    #[cfg(feature = "cuda")]
    pub(crate) fn decode_step_from_device_into(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        hidden_fp8: &ferrule_cuda::context::CudaPreparedFp8Activation<'_>,
        position: usize,
        operators: &mut DeepSeekV4OperatorContext,
        arena: &mut DeepSeekV4AttentionDecodeArena,
    ) -> Result<()> {
        let cfg = self.config;
        if hidden_dev.len() != cfg.hidden_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} attention device input mismatch: expected {}, got {}",
                self.layer,
                cfg.hidden_size,
                hidden_dev.len()
            )));
        }
        if cfg.compress_ratio == 0 {
            return self.decode_step_no_compress_from_device_into(
                &mut cache.window,
                hidden_fp8,
                position,
                operators,
                cfg,
                arena,
            );
        }
        let compressed = self.compressed.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {} has compress_ratio {} but no typed compressed attention payload is bound",
                self.layer, cfg.compress_ratio
            ))
        })?;
        let rope = cfg.rope_params();
        let position_i32 = i32::try_from(position)
            .map_err(|_| Error::Model("DeepSeek-V4 decode position exceeds i32 ABI".into()))?;
        operators.cuda_mut()?.ops.fill_i32_sequence_prefix(
            arena.positions.device_mut_invalidate_host(),
            position_i32,
            1,
        )?;
        self.project_decode_rows_from_device_into(
            hidden_dev, hidden_fp8, position, operators, arena,
        )?;
        self.decode_step_compressed_projected_continuation(
            cache, hidden_dev, position, operators, cfg, compressed, rope, arena,
        )?;
        self.project_decode_context_rows_from_device_into(position, operators, arena)
    }

    #[cfg(feature = "cuda")]
    fn project_decode_rows_from_device_into(
        &self,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        hidden_fp8: &ferrule_cuda::context::CudaPreparedFp8Activation<'_>,
        max_position: usize,
        operators: &mut DeepSeekV4OperatorContext,
        arena: &mut DeepSeekV4AttentionDecodeArena,
    ) -> Result<()> {
        let cfg = self.config;
        let rows = arena.positions.len();
        if rows == 0 || hidden_dev.len() != rows * cfg.hidden_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} packed attention input mismatch: rows={rows} expected={} got={}",
                self.layer,
                rows * cfg.hidden_size,
                hidden_dev.len()
            )));
        }
        let required_positions = max_position
            .checked_add(1)
            .ok_or_else(|| Error::Model("DeepSeek-V4 packed RoPE position overflow".into()))?;
        let layer_tag = format!("attn_L{}", self.layer);
        let rope_name = format!("rope_{layer_tag}");
        operators.record_attention_call(self.layer, rows);
        operators.cuda_mut()?.ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            required_positions,
        )?;

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.query_a_kv_from_prepared_fp8_into(
            self.layer,
            hidden_fp8,
            &mut arena.q_latent,
            &mut arena.kv_raw,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qa,
            stage_start,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_q_latent",
            &arena.q_latent,
            cfg.q_lora_rank,
        )?;

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_layer_rows_device_into(
            self.layer,
            DeepSeekV4CudaLayerNorm::Query,
            &arena.q_latent,
            rows,
            cfg.norm_eps,
            &mut arena.q_norm,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QNorm,
            stage_start,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_q_norm",
            &arena.q_norm,
            cfg.q_lora_rank,
        )?;
        if self
            .compressed
            .as_ref()
            .is_some_and(|value| value.indexer.is_some())
        {
            operators
                .cuda_mut()?
                .ops
                .copy_f32_into_slot(&arena.q_norm, &mut arena.q_indexer, 0)?;
        }

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.linear_rows_from_device_into(
            self.layer,
            DeepSeekV4CudaLinear::QueryB,
            &arena.q_norm,
            rows,
            &mut arena.query_raw,
            &mut arena.linear_workspace,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qb,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_heads_from_device_into(
            &arena.query_raw,
            rows * cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
            &mut arena.query,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QHeadNorm,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rope_tail_rows_indexed_from_device(
            &rope_name,
            &mut arena.query,
            &arena.positions,
            max_position,
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
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_query",
            &arena.query,
            cfg.q_full_dim(),
        )?;

        let stage_start = operators.profile_start();
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvProj,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_layer_rows_device_into(
            self.layer,
            DeepSeekV4CudaLayerNorm::KeyValue,
            &arena.kv_raw,
            rows,
            cfg.norm_eps,
            &mut arena.kv,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvNorm,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rope_tail_rows_indexed_from_device(
            &rope_name,
            &mut arena.kv,
            &arena.positions,
            max_position,
            1,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        operators
            .cuda_mut()?
            .ops
            .fp8_attention_kv_qat_quantize_buffer_in_place(
                &mut arena.kv,
                cfg.head_dim,
                cfg.rope_head_dim,
            )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvRopeQuant,
            stage_start,
        )?;
        operators.capture_parity_checkpoint_rows(self.layer, "attn_kv", &arena.kv, cfg.head_dim)
    }

    #[cfg(feature = "cuda")]
    fn project_decode_context_rows_from_device_into(
        &self,
        max_position: usize,
        operators: &mut DeepSeekV4OperatorContext,
        arena: &mut DeepSeekV4AttentionDecodeArena,
    ) -> Result<()> {
        let cfg = self.config;
        let rows = arena.positions.len();
        let rope_name = format!("rope_attn_L{}", self.layer);
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rope_tail_rows_indexed_from_device(
            &rope_name,
            &mut arena.context,
            &arena.positions,
            max_position,
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
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_context_rope",
            &arena.context,
            cfg.q_full_dim(),
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.mla_output_rows_from_device_into(
            &arena.context,
            rows,
            cfg,
            self.layer,
            &mut arena.latent,
            &mut arena.linear_workspace,
            &mut arena.output,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::OutputB,
            stage_start,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_output_a",
            &arena.latent,
            self.payload.output_b.format.in_features(),
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_output",
            &arena.output,
            cfg.hidden_size,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn packed_rows_from_device_into(
        &self,
        caches: &mut [&mut DeepSeekV4AttentionCache],
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        hidden_fp8: &ferrule_cuda::context::CudaPreparedFp8Activation<'_>,
        positions: &[usize],
        row_to_sequence: &[usize],
        sequence_major_rows: &[usize],
        sequence_phases: &[ForwardPhase],
        paged_bindings: &[DeepSeekV4PagedKvBinding],
        operators: &mut DeepSeekV4OperatorContext,
        arena: &mut DeepSeekV4AttentionDecodeArena,
        transition: &mut DeepSeekV4AttentionRowsTransitionArena,
    ) -> Result<()> {
        let rows = positions.len();
        if rows == 0
            || row_to_sequence.len() != rows
            || sequence_major_rows.len() != rows
            || caches.len() != paged_bindings.len()
            || sequence_phases.len() != caches.len()
            || row_to_sequence
                .iter()
                .any(|sequence| *sequence >= caches.len())
        {
            return Err(Error::Model(
                "DeepSeek-V4 packed attention row/sequence metadata is inconsistent".into(),
            ));
        }

        let visible_lens = positions
            .iter()
            .map(|position| {
                position
                    .checked_add(1)
                    .and_then(|length| i32::try_from(length).ok())
                    .ok_or_else(|| {
                        Error::Model("packed attention visible length exceeds i32 ABI".into())
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if self.config.compress_ratio == 0 {
            let cfg = self.config;
            let max_position = positions
                .iter()
                .copied()
                .max()
                .ok_or_else(|| Error::Model("packed decode positions are empty".into()))?;
            let positions_i32 = decode_metadata_i32(positions, "position")?;
            {
                let ops = &operators.cuda_mut()?.ops;
                ops.update_i32_host_mirror(&positions_i32, &mut arena.positions)?;
                ops.update_i32_host_mirror(&visible_lens, &mut arena.visible_lens)?;
            }
            self.project_decode_rows_from_device_into(
                hidden_dev,
                hidden_fp8,
                max_position,
                operators,
                arena,
            )?;
            let zero_kv = vec![0.0f32; cfg.head_dim];
            for &row in sequence_major_rows {
                let sequence = row_to_sequence[row];
                caches[sequence].window.append(positions[row], &zero_kv)?;
            }
            operators.cuda_mut()?.paged_scatter_rows_from_device(
                0,
                self.layer,
                &arena.kv,
                &arena.positions,
                None,
                cfg.head_dim,
            )?;
            let attention_topk = positions
                .iter()
                .map(|position| position.saturating_add(1).min(cfg.window_size))
                .max()
                .ok_or_else(|| Error::Model("packed attention positions are empty".into()))?;
            let stage_start = operators.profile_start();
            operators
                .cuda_mut()?
                .paged_window_sparse_attention_rows_into(
                    &arena.query,
                    positions,
                    &arena.visible_lens,
                    rows,
                    self.layer,
                    SparseAttentionSpec {
                        heads: cfg.num_heads,
                        head_dim: cfg.head_dim,
                        topk: attention_topk,
                        softmax_scale: (cfg.head_dim as f32).powf(-0.5),
                        has_attention_sink: !self.payload.attention_sink.is_empty(),
                    },
                    &mut arena.context,
                )?;
            record_attention_stage(
                operators,
                self.layer,
                DeepSeekV4AttentionProfileStage::SparseAttention,
                stage_start,
            )?;
            return self.project_decode_context_rows_from_device_into(
                max_position,
                operators,
                arena,
            );
        }

        let compressed = self.compressed.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {} has compressed config without payload",
                self.layer
            ))
        })?;
        let cfg = self.config;
        let rope = cfg.rope_params();
        let max_position = positions
            .iter()
            .copied()
            .max()
            .ok_or_else(|| Error::Model("packed decode positions are empty".into()))?;
        let positions_i32 = decode_metadata_i32(positions, "position")?;
        {
            let ops = &operators.cuda_mut()?.ops;
            ops.update_i32_host_mirror(&positions_i32, &mut arena.positions)?;
            ops.update_i32_host_mirror(&visible_lens, &mut arena.visible_lens)?;
        }
        self.project_decode_rows_from_device_into(
            hidden_dev,
            hidden_fp8,
            max_position,
            operators,
            arena,
        )?;
        if sequence_phases.contains(&ForwardPhase::Prefill) {
            if compressed.indexer.is_some() {
                let projected = arena
                    .indexer_compressor
                    .as_mut()
                    .expect("indexer compressor arena exists");
                operators.cuda_mut()?.compressor_rows_from_device_into(
                    self.layer,
                    DeepSeekV4CudaCompressor::Indexer,
                    hidden_dev,
                    rows,
                    &mut projected.kv,
                    &mut projected.score,
                )?;
            }
            let projected = arena
                .main_compressor
                .as_mut()
                .expect("main compressor arena exists");
            operators.cuda_mut()?.compressor_rows_from_device_into(
                self.layer,
                DeepSeekV4CudaCompressor::Main,
                hidden_dev,
                rows,
                &mut projected.kv,
                &mut projected.score,
            )?;
        }
        (|| -> Result<()> {
            let zero_kv = vec![0.0f32; cfg.head_dim];
            operators.cuda_mut()?.paged_scatter_rows_from_device(
                0,
                self.layer,
                &arena.kv,
                &arena.positions,
                None,
                cfg.head_dim,
            )?;

            let mut main_positions = vec![0i32; rows];
            let mut main_mask = vec![0i32; rows];
            let mut indexer_positions = vec![0i32; rows];
            let mut indexer_mask = vec![0i32; rows];
            let mut window_lens = vec![0usize; rows];
            let mut compressed_lens = vec![0usize; rows];
            for &row in sequence_major_rows {
                let sequence = row_to_sequence[row];
                let cache = &mut caches[sequence];
                cache.window.append(positions[row], &zero_kv)?;
                operators
                    .cuda_mut()?
                    .ops
                    .copy_f32_range(
                        hidden_dev,
                        row * cfg.hidden_size,
                        &mut transition.input,
                        0,
                        cfg.hidden_size,
                    )
                    .map_err(|error| {
                        Error::Internal(format!(
                            "DeepSeek-V4 layer {} row {row} transition input copy failed: source_len={} destination_len={}: {error}",
                            self.layer,
                            hidden_dev.len(),
                            transition.input.len()
                        ))
                    })?;

                if let Some(indexer) = compressed.indexer.as_ref() {
                    let new_indexer_kv = {
                        let (compressor_state, window) =
                            (&mut cache.indexer_compressor, &mut cache.window);
                        let state = compressor_state.as_mut().ok_or_else(|| {
                            Error::Model(format!(
                                "DeepSeek-V4 layer {} missing indexer compressor state",
                                self.layer
                            ))
                        })?;
                        let scratch = transition.indexer_compressor.as_mut().ok_or_else(|| {
                            Error::Internal(format!(
                                "DeepSeek-V4 layer {} missing indexer compressor row arena",
                                self.layer
                            ))
                        })?;
                        let cuda = window.cuda_state_mut();
                        if sequence_phases[sequence] == ForwardPhase::Prefill {
                            let projected = arena
                                .indexer_compressor
                                .as_ref()
                                .expect("indexer compressor arena exists");
                            state.append_projected_step_from_device_into(
                                &indexer.compressor,
                                self.layer,
                                DeepSeekV4CudaCompressor::Indexer,
                                &projected.kv,
                                &projected.score,
                                row,
                                positions[row],
                                cfg.rope_head_dim,
                                rope,
                                &format!("rope_indexer_compress_L{}", self.layer),
                                &mut cuda.indexer_compressor_recurrent,
                                &mut cuda.indexer_compressor_needs_reset,
                                operators,
                                scratch,
                            )?
                        } else {
                            state.append_step_from_device_into(
                                &indexer.compressor,
                                self.layer,
                                DeepSeekV4CudaCompressor::Indexer,
                                &transition.input,
                                positions[row],
                                cfg.rope_head_dim,
                                rope,
                                &format!("rope_indexer_compress_L{}", self.layer),
                                &mut cuda.indexer_compressor_recurrent,
                                &mut cuda.indexer_compressor_needs_reset,
                                operators,
                                scratch,
                            )?
                        }
                    };
                    operators.fail_compressor_transition_if_armed(true)?;
                    if new_indexer_kv {
                        let index = cache.indexer_compressed_len(cfg.index_head_dim);
                        cache
                            .indexer_compressed
                            .resize(cache.indexer_compressed.len() + cfg.index_head_dim, 0.0);
                        indexer_positions[row] = i32::try_from(index).map_err(|_| {
                            Error::Model("packed indexer position exceeds i32 ABI".into())
                        })?;
                        indexer_mask[row] = 1;
                        let transition_normalized = &transition
                            .indexer_compressor
                            .as_ref()
                            .expect("indexer compressor row arena exists")
                            .normalized;
                        let packed_normalized = &mut arena
                            .indexer_compressor
                            .as_mut()
                            .expect("indexer compressor arena exists")
                            .normalized;
                        operators
                            .cuda_mut()?
                            .ops
                            .copy_f32_range(
                                transition_normalized,
                                0,
                                packed_normalized,
                                row * cfg.index_head_dim,
                                cfg.index_head_dim,
                            )
                            .map_err(|error| {
                                Error::Internal(format!(
                                    "DeepSeek-V4 layer {} row {row} indexer normalized copy failed: source_len={} destination_len={} row_dim={}: {error}",
                                    self.layer,
                                    transition_normalized.len(),
                                    packed_normalized.len(),
                                    cfg.index_head_dim
                                ))
                            })?;
                    }
                }

                let new_main_kv = {
                    let (compressor_state, window) =
                        (&mut cache.main_compressor, &mut cache.window);
                    let state = compressor_state.as_mut().ok_or_else(|| {
                        Error::Model(format!(
                            "DeepSeek-V4 layer {} missing main compressor state",
                            self.layer
                        ))
                    })?;
                    let scratch = transition.main_compressor.as_mut().ok_or_else(|| {
                        Error::Internal(format!(
                            "DeepSeek-V4 layer {} missing main compressor row arena",
                            self.layer
                        ))
                    })?;
                    let cuda = window.cuda_state_mut();
                    if sequence_phases[sequence] == ForwardPhase::Prefill {
                        let projected = arena
                            .main_compressor
                            .as_ref()
                            .expect("main compressor arena exists");
                        state.append_projected_step_from_device_into(
                            &compressed.compressor,
                            self.layer,
                            DeepSeekV4CudaCompressor::Main,
                            &projected.kv,
                            &projected.score,
                            row,
                            positions[row],
                            cfg.rope_head_dim,
                            rope,
                            &format!("rope_main_compress_L{}", self.layer),
                            &mut cuda.main_compressor_recurrent,
                            &mut cuda.main_compressor_needs_reset,
                            operators,
                            scratch,
                        )?
                    } else {
                        state.append_step_from_device_into(
                            &compressed.compressor,
                            self.layer,
                            DeepSeekV4CudaCompressor::Main,
                            &transition.input,
                            positions[row],
                            cfg.rope_head_dim,
                            rope,
                            &format!("rope_main_compress_L{}", self.layer),
                            &mut cuda.main_compressor_recurrent,
                            &mut cuda.main_compressor_needs_reset,
                            operators,
                            scratch,
                        )?
                    }
                };
                operators.fail_compressor_transition_if_armed(false)?;
                if new_main_kv {
                    let index = cache.compressed_len();
                    cache
                        .compressed
                        .resize(cache.compressed.len() + cfg.head_dim, 0.0);
                    main_positions[row] = i32::try_from(index).map_err(|_| {
                        Error::Model("packed main compressed position exceeds i32 ABI".into())
                    })?;
                    main_mask[row] = 1;
                    let transition_normalized = &transition
                        .main_compressor
                        .as_ref()
                        .expect("main compressor row arena exists")
                        .normalized;
                    let packed_normalized = &mut arena
                        .main_compressor
                        .as_mut()
                        .expect("main compressor arena exists")
                        .normalized;
                    operators
                        .cuda_mut()?
                        .ops
                        .copy_f32_range(
                            transition_normalized,
                            0,
                            packed_normalized,
                            row * cfg.head_dim,
                            cfg.head_dim,
                        )
                        .map_err(|error| {
                            Error::Internal(format!(
                                "DeepSeek-V4 layer {} row {row} main normalized copy failed: source_len={} destination_len={} row_dim={}: {error}",
                                self.layer,
                                transition_normalized.len(),
                                packed_normalized.len(),
                                cfg.head_dim
                            ))
                        })?;
                }
                window_lens[row] = cache.window.len;
                if compressed.indexer.is_some() {
                    compressed_lens[row] = cache.indexer_compressed_len(cfg.index_head_dim);
                }
            }

            {
                let ops = &operators.cuda_mut()?.ops;
                ops.update_i32_host_mirror(&main_positions, &mut arena.main_positions)?;
                ops.update_i32_host_mirror(&main_mask, &mut arena.main_mask)?;
                ops.update_i32_host_mirror(&indexer_positions, &mut arena.indexer_positions)?;
                ops.update_i32_host_mirror(&indexer_mask, &mut arena.indexer_mask)?;
            }
            operators.cuda_mut()?.paged_scatter_rows_from_device(
                1,
                self.layer,
                &arena
                    .main_compressor
                    .as_ref()
                    .expect("main compressor arena exists")
                    .normalized,
                &arena.main_positions,
                Some(&arena.main_mask),
                cfg.head_dim,
            )?;
            if compressed.indexer.is_some() {
                operators.cuda_mut()?.paged_scatter_rows_from_device(
                    2,
                    self.layer,
                    &arena
                        .indexer_compressor
                        .as_ref()
                        .expect("indexer compressor arena exists")
                        .normalized,
                    &arena.indexer_positions,
                    Some(&arena.indexer_mask),
                    cfg.index_head_dim,
                )?;
            }

            if compressed.indexer.is_some() {
                operators.cuda_mut()?.linear_rows_from_device_into(
                    self.layer,
                    DeepSeekV4CudaLinear::IndexerQuery,
                    &arena.q_indexer,
                    rows,
                    &mut arena.index_query,
                    &mut arena.linear_workspace,
                )?;
                let index_rope_dim = cfg.rope_head_dim.min(cfg.index_head_dim);
                let index_rope_name = format!("rope_indexer_query_L{}", self.layer);
                let required_positions = positions
                    .iter()
                    .copied()
                    .max()
                    .and_then(|position| position.checked_add(1))
                    .ok_or_else(|| Error::Model("packed indexer RoPE position overflow".into()))?;
                operators.cuda_mut()?.ensure_rope_tables_with_params(
                    &index_rope_name,
                    index_rope_dim,
                    cfg.rope_params(),
                    required_positions,
                )?;
                operators.cuda_mut()?.rope_tail_rows_indexed_from_device(
                    &index_rope_name,
                    &mut arena.index_query,
                    &arena.positions,
                    max_position,
                    cfg.index_n_heads as u32,
                    cfg.index_head_dim as u32,
                    index_rope_dim as u32,
                    false,
                )?;
                operators
                    .cuda_mut()?
                    .ops
                    .fp4_hadamard_qat_quantize_buffer_in_place(
                        &mut arena.index_query,
                        cfg.index_head_dim,
                    )?;
                operators.cuda_mut()?.linear_rows_from_device_into(
                    self.layer,
                    DeepSeekV4CudaLinear::IndexerWeights,
                    hidden_dev,
                    rows,
                    &mut arena.index_weights,
                    &mut arena.linear_workspace,
                )?;
            }
            let window_lens_i32 = decode_metadata_i32(&window_lens, "window length")?;
            let compressed_lens_i32 = decode_metadata_i32(&compressed_lens, "compressed length")?;
            {
                let ops = &operators.cuda_mut()?.ops;
                ops.update_i32_host_mirror(&window_lens_i32, &mut arena.window_lens)?;
                ops.update_i32_host_mirror(&compressed_lens_i32, &mut arena.compressed_lens)?;
            }
            let weight_scale =
                (cfg.index_head_dim as f32).powf(-0.5) * (cfg.index_n_heads as f32).powf(-0.5);
            operators
                .cuda_mut()?
                .decode_topk_indices_paged_indexer_rows_into(
                    &arena.index_query,
                    &arena.index_weights,
                    &arena.positions,
                    &arena.window_lens,
                    &arena.compressed_lens,
                    self.layer,
                    cfg.window_size,
                    cfg.index_topk,
                    cfg.index_n_heads,
                    cfg.index_head_dim,
                    weight_scale,
                    &mut arena.topk,
                    &mut arena.topk_selectors,
                )?;
            operators
                .cuda_mut()?
                .dual_plane_paged_sparse_attention_rows_into(
                    &arena.query,
                    &arena.topk,
                    &arena.topk_selectors,
                    &arena.visible_lens,
                    rows,
                    self.layer,
                    cfg.sparse_spec_with_topk(cfg.window_size + cfg.index_topk),
                    &mut arena.context,
                )?;
            operators.capture_parity_checkpoint_last_row(
                self.layer,
                "attn_context",
                &arena.context,
                cfg.q_full_dim(),
            )?;
            Ok(())
        })()?;
        self.project_decode_context_rows_from_device_into(max_position, operators, arena)
    }

    /// CUDA fully device-resident compressed attention path.
    ///
    /// Accepts the hidden state as a `CudaF32Buffer` and returns the output as a
    /// `CudaF32Buffer`, eliminating
    /// the D2H+H2D round-trip at the call boundary (4 syncs → 0 in steady state).
    ///
    /// Remaining host syncs are now limited to compressor CPU-shadow state
    /// maintenance and MoE/router bookkeeping outside this attention function.
    /// q-latent/hidden downloads and top-k uploads have been removed from this
    /// decode path; indexer compressed KV is served from the runtime page pool.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn decode_step_compressed_projected_continuation(
        &self,
        cache: &mut DeepSeekV4AttentionCache,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        operators: &mut DeepSeekV4OperatorContext,
        cfg: DeepSeekV4AttentionConfig,
        compressed: &DeepSeekV4CompressedAttentionPayload,
        rope: DeepSeekV4RopeParams,
        arena: &mut DeepSeekV4AttentionDecodeArena,
    ) -> Result<()> {
        let rope_positions = required_rope_positions(position, 1, 1)?;
        // Append to device combined KV cache (no D2H/H2D round-trip).
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.paged_write_rows(
            0,
            self.layer,
            &arena.kv,
            position,
            1,
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
            let stage_start = operators.profile_start();
            let new_indexer_kv = {
                let (compressor_state, window) = (&mut cache.indexer_compressor, &mut cache.window);
                let state = compressor_state.as_mut().ok_or_else(|| {
                    Error::Model(format!(
                        "DeepSeek-V4 layer {} missing indexer compressor state",
                        self.layer
                    ))
                })?;
                let scratch = arena.indexer_compressor.as_mut().ok_or_else(|| {
                    Error::Internal(format!(
                        "DeepSeek-V4 layer {} missing indexer compressor arena",
                        self.layer
                    ))
                })?;
                let cuda = window.cuda_state_mut();
                state.append_step_from_device_into(
                    &indexer.compressor,
                    self.layer,
                    DeepSeekV4CudaCompressor::Indexer,
                    hidden_dev,
                    position,
                    cfg.rope_head_dim,
                    rope,
                    &format!("rope_indexer_compress_L{}", self.layer),
                    &mut cuda.indexer_compressor_recurrent,
                    &mut cuda.indexer_compressor_needs_reset,
                    operators,
                    scratch,
                )?
            };
            record_attention_stage(
                operators,
                self.layer,
                DeepSeekV4AttentionProfileStage::IndexerCompress,
                stage_start,
            )?;
            operators.fail_compressor_transition_if_armed(true)?;
            if new_indexer_kv {
                let compressed_index = cache.indexer_compressed_len(cfg.index_head_dim);
                cache
                    .indexer_compressed
                    .resize(cache.indexer_compressed.len() + cfg.index_head_dim, 0.0);
                operators.cuda_mut()?.paged_write_rows(
                    2,
                    self.layer,
                    &arena
                        .indexer_compressor
                        .as_ref()
                        .expect("indexer compressor arena exists")
                        .normalized,
                    compressed_index,
                    1,
                    cfg.index_head_dim,
                )?;
            }
        }

        let stage_start = operators.profile_start();
        let new_main_kv = {
            let (compressor_state, window) = (&mut cache.main_compressor, &mut cache.window);
            let state = compressor_state.as_mut().ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 layer {} missing main compressor state",
                    self.layer
                ))
            })?;
            let scratch = arena.main_compressor.as_mut().ok_or_else(|| {
                Error::Internal(format!(
                    "DeepSeek-V4 layer {} missing main compressor arena",
                    self.layer
                ))
            })?;
            let cuda = window.cuda_state_mut();
            state.append_step_from_device_into(
                &compressed.compressor,
                self.layer,
                DeepSeekV4CudaCompressor::Main,
                hidden_dev,
                position,
                cfg.rope_head_dim,
                rope,
                &format!("rope_main_compress_L{}", self.layer),
                &mut cuda.main_compressor_recurrent,
                &mut cuda.main_compressor_needs_reset,
                operators,
                scratch,
            )?
        };
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::MainCompress,
            stage_start,
        )?;
        operators.fail_compressor_transition_if_armed(false)?;
        if new_main_kv {
            cache
                .compressed
                .resize(cache.compressed.len() + cfg.head_dim, 0.0);
        }

        // Combined KV cache window append already done above (device path).
        // Only compressed KV append remains here.
        if new_main_kv {
            let stage_start = operators.profile_start();
            let compressed_index = cache.compressed_len().saturating_sub(1);
            operators.cuda_mut()?.paged_write_rows(
                1,
                self.layer,
                &arena
                    .main_compressor
                    .as_ref()
                    .expect("main compressor arena exists")
                    .normalized,
                compressed_index,
                1,
                cfg.head_dim,
            )?;
            record_attention_stage(
                operators,
                self.layer,
                DeepSeekV4AttentionProfileStage::CompressedKvUpload,
                stage_start,
            )?;
        }

        let stage_start = operators.profile_start();
        let window_len = cache.window.len;
        let topk_len = if compressed.indexer.is_some() {
            let compressed_len = cache.indexer_compressed_len(cfg.index_head_dim);
            let indexer_cols = cfg.index_topk.min(compressed_len);
            if indexer_cols == 0 {
                operators
                    .cuda_mut()?
                    .ops
                    .fill_dsv4_decode_attention_topk_into(
                        &mut arena.topk,
                        position,
                        cfg.window_size,
                        window_len,
                        0,
                    )?
            } else {
                operators.cuda_mut()?.linear_matvec_from_device_into(
                    self.layer,
                    DeepSeekV4CudaLinear::IndexerQuery,
                    &mut arena.q_indexer,
                    &mut arena.index_query,
                )?;
                sync_attention_profile_checkpoint(
                    operators,
                    self.layer,
                    "decode_index_query_projection",
                )?;
                let index_rope_dim = cfg.rope_head_dim.min(cfg.index_head_dim);
                let index_rope_name = format!("rope_indexer_query_L{}", self.layer);
                operators.cuda_mut()?.ensure_rope_tables_with_params(
                    &index_rope_name,
                    index_rope_dim,
                    cfg.rope_params(),
                    rope_positions,
                )?;
                sync_attention_profile_checkpoint(
                    operators,
                    self.layer,
                    "decode_index_rope_and_kv_cache_prepare",
                )?;
                operators
                    .cuda_mut()?
                    .ops
                    .copy_f32_into_slot(hidden_dev, &mut arena.hidden_a, 0)?;
                operators.cuda_mut()?.linear_matvec_from_device_into(
                    self.layer,
                    DeepSeekV4CudaLinear::IndexerWeights,
                    &mut arena.hidden_a,
                    &mut arena.index_weights,
                )?;
                let weight_scale =
                    (cfg.index_head_dim as f32).powf(-0.5) * (cfg.index_n_heads as f32).powf(-0.5);
                if dsv4_fused_indexer_decode_topk_supported(operators, cfg, index_rope_dim) {
                    operators
                        .cuda_mut()?
                        .decode_topk_indices_fused_index_query_paged_indexer_into(
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
                    operators.cuda_mut()?.rope_tail_from_device(
                        &index_rope_name,
                        &mut arena.index_query,
                        position as u32,
                        cfg.index_n_heads as u32,
                        cfg.index_head_dim as u32,
                        index_rope_dim as u32,
                        false,
                    )?;
                    operators
                        .cuda_mut()?
                        .ops
                        .fp4_hadamard_qat_quantize_buffer_in_place(
                            &mut arena.index_query,
                            cfg.index_head_dim,
                        )?;
                    operators
                        .cuda_mut()?
                        .decode_topk_indices_from_paged_indexer_into(
                            &arena.index_query,
                            &arena.index_weights,
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
                            &mut arena.topk,
                        )?;
                }
                cfg.window_size + indexer_cols
            }
        } else {
            let compressed_len = cache.compressed_len();
            operators
                .cuda_mut()?
                .ops
                .fill_dsv4_decode_attention_topk_into(
                    &mut arena.topk,
                    position,
                    cfg.window_size,
                    window_len,
                    compressed_len,
                )?
        };
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::TopkBuild,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        operators
            .cuda_mut()?
            .sparse_attention_with_paged_kv_topk_into(
                &arena.query,
                self.layer,
                position,
                cfg.window_size,
                &arena.topk,
                1,
                cfg.sparse_spec_with_topk(topk_len),
                &mut arena.context,
            )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::SparseAttention,
            stage_start,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attn_context",
            &arena.context,
            cfg.q_full_dim(),
        )?;
        Ok(())
    }

    /// Execute the official DSV4 attention path for non-compressed layers
    /// (`compress_ratio == 0`, i.e. local sliding-window MLA only).
    pub fn decode_step_no_compress(
        &self,
        cache: &mut DeepSeekV4WindowKvCache,
        hidden: &[f32],
        position: usize,
    ) -> Result<Vec<f32>> {
        let mut operators = DeepSeekV4OperatorContext::new_cpu()?;
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
    /// Accepts and returns `CudaF32Buffer`, eliminating the boundary D2H+H2D
    /// round-trip.
    #[cfg(feature = "cuda")]
    fn decode_step_no_compress_from_device_into(
        &self,
        cache: &mut DeepSeekV4WindowKvCache,
        hidden_fp8: &ferrule_cuda::context::CudaPreparedFp8Activation<'_>,
        position: usize,
        operators: &mut DeepSeekV4OperatorContext,
        cfg: DeepSeekV4AttentionConfig,
        arena: &mut DeepSeekV4AttentionDecodeArena,
    ) -> Result<()> {
        let layer_tag = format!("attn_L{}", self.layer);
        let rope_name = format!("rope_{layer_tag}");
        operators.record_attention_call(self.layer, 1);

        let rope_positions = required_rope_positions(position, 1, 1)?;
        operators.cuda_mut()?.ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            rope_positions,
        )?;
        // QueryA and KV share the HC producer's FP8 activation.
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.query_a_kv_from_prepared_fp8_into(
            self.layer,
            hidden_fp8,
            &mut arena.q_latent,
            &mut arena.kv_raw,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qa,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_layer_device_into(
            self.layer,
            DeepSeekV4CudaLayerNorm::Query,
            &arena.q_latent,
            cfg.norm_eps,
            &mut arena.q_norm,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QNorm,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.linear_matvec_from_device_into(
            self.layer,
            DeepSeekV4CudaLinear::QueryB,
            &mut arena.q_norm,
            &mut arena.query_raw,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qb,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_heads_from_device_into(
            &arena.query_raw,
            cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
            &mut arena.query,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QHeadNorm,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rope_tail_from_device(
            &rope_name,
            &mut arena.query,
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

        // Normalize and rotate the KV output produced by the semantic bundle.
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_layer_device_into(
            self.layer,
            DeepSeekV4CudaLayerNorm::KeyValue,
            &arena.kv_raw,
            cfg.norm_eps,
            &mut arena.kv,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvNorm,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rope_tail_from_device(
            &rope_name,
            &mut arena.kv,
            position as u32,
            1,
            cfg.head_dim as u32,
            cfg.rope_head_dim as u32,
            false,
        )?;
        operators
            .cuda_mut()?
            .ops
            .fp8_attention_kv_qat_quantize_buffer_in_place(
                &mut arena.kv,
                cfg.head_dim,
                cfg.rope_head_dim,
            )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvRopeQuant,
            stage_start,
        )?;
        // Append to device KV cache.
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.paged_write_rows(
            0,
            self.layer,
            &arena.kv,
            position,
            1,
            cfg.head_dim,
        )?;
        cache.record_device_append();
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvCacheAppend,
            stage_start,
        )?;

        // Sparse attention on device.
        let shape = ferrule_cuda::transformer::sparse_attention::CudaSparseAttentionShape {
            batch_size: 1,
            tokens_per_batch: 1,
            kv_len: cfg.window_size,
            heads: cfg.num_heads,
            head_dim: cfg.head_dim,
            topk: cfg.window_size,
            softmax_scale: (cfg.head_dim as f32).powf(-0.5),
        };
        let stage_start = operators.profile_start();
        operators
            .cuda_mut()?
            .sparse_attention_topk_from_device_into(
                &arena.query,
                self.layer,
                position,
                cfg.window_size,
                shape,
                &mut arena.context,
            )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::SparseAttention,
            stage_start,
        )?;

        // Inverse rotary on context (device).
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rope_tail_from_device(
            &rope_name,
            &mut arena.context,
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

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.mla_output_rows_from_device_into(
            &arena.context,
            1,
            cfg,
            self.layer,
            &mut arena.latent,
            &mut arena.linear_workspace,
            &mut arena.output,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::OutputB,
            stage_start,
        )?;
        Ok(())
    }

    /// DSpark proposal attention is deliberately separate from ordinary packed
    /// target attention: all five rows see the complete ephemeral block, and the
    /// block KV is never appended to the committed page table.
    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    pub(crate) fn dspark_proposal_block_from_device_into(
        &self,
        stage: usize,
        hidden_fp8: &ferrule_cuda::context::CudaPreparedFp8Activation<'_>,
        sequence_tokens: usize,
        operators: &mut DeepSeekV4OperatorContext,
        arena: &mut DeepSeekV4AttentionDecodeArena,
        dspark: &mut DeepSeekV4DsparkAttentionBuffers,
    ) -> Result<()> {
        let cfg = self.config;
        let rows = ferrule_cuda::cutlass::DSPARK_PROPOSAL_ROWS;
        if sequence_tokens == 0
            || cfg.compress_ratio != 0
            || arena.query.len() != rows.saturating_mul(cfg.q_full_dim())
            || arena.kv.len() != rows.saturating_mul(cfg.head_dim)
            || arena.context.len() != rows.saturating_mul(cfg.q_full_dim())
        {
            return Err(Error::Model(format!(
                "DeepSeek-V4 DSpark proposal-attention shape mismatch at stage {stage}: sequence_tokens={sequence_tokens} compress_ratio={} query={} kv={} context={} rows={rows}",
                cfg.compress_ratio,
                arena.query.len(),
                arena.kv.len(),
                arena.context.len()
            )));
        }
        let required_positions = sequence_tokens
            .checked_add(rows)
            .ok_or_else(|| Error::Model("DeepSeek-V4 DSpark proposal position overflow".into()))?;
        let rope_name = format!("rope_dspark_stage_{stage}");
        operators.cuda_mut()?.ensure_rope_tables_with_params(
            &rope_name,
            cfg.rope_head_dim,
            cfg.rope_params(),
            required_positions,
        )?;
        operators.record_attention_call(self.layer, rows);

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.query_a_kv_from_prepared_fp8_into(
            self.layer,
            hidden_fp8,
            &mut arena.q_latent,
            &mut arena.kv_raw,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qa,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_layer_rows_device_into(
            self.layer,
            DeepSeekV4CudaLayerNorm::Query,
            &arena.q_latent,
            rows,
            cfg.norm_eps,
            &mut arena.q_norm,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QNorm,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.linear_rows_from_device_into(
            self.layer,
            DeepSeekV4CudaLinear::QueryB,
            &arena.q_norm,
            rows,
            &mut arena.query_raw,
            &mut arena.linear_workspace,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::Qb,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_heads_from_device_into(
            &arena.query_raw,
            rows * cfg.num_heads,
            cfg.head_dim,
            cfg.norm_eps,
            &mut arena.query,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::QHeadNorm,
            stage_start,
        )?;
        operators.cuda_mut()?.rope_tail_rows_from_device(
            &rope_name,
            &mut arena.query,
            u32::try_from(sequence_tokens).map_err(|_| {
                Error::Model("DeepSeek-V4 DSpark proposal position exceeds u32".into())
            })?,
            u32::try_from(rows)
                .map_err(|_| Error::Model("DeepSeek-V4 DSpark row count exceeds u32".into()))?,
            u32::try_from(cfg.num_heads)
                .map_err(|_| Error::Model("DeepSeek-V4 DSpark head count exceeds u32".into()))?,
            u32::try_from(cfg.head_dim)
                .map_err(|_| Error::Model("DeepSeek-V4 DSpark head dim exceeds u32".into()))?,
            u32::try_from(cfg.rope_head_dim)
                .map_err(|_| Error::Model("DeepSeek-V4 DSpark RoPE dim exceeds u32".into()))?,
            false,
        )?;

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.rms_norm_layer_rows_device_into(
            self.layer,
            DeepSeekV4CudaLayerNorm::KeyValue,
            &arena.kv_raw,
            rows,
            cfg.norm_eps,
            &mut arena.kv,
        )?;
        operators.cuda_mut()?.rope_tail_rows_from_device(
            &rope_name,
            &mut arena.kv,
            u32::try_from(sequence_tokens).map_err(|_| {
                Error::Model("DeepSeek-V4 DSpark proposal position exceeds u32".into())
            })?,
            u32::try_from(rows)
                .map_err(|_| Error::Model("DeepSeek-V4 DSpark row count exceeds u32".into()))?,
            1,
            u32::try_from(cfg.head_dim)
                .map_err(|_| Error::Model("DeepSeek-V4 DSpark head dim exceeds u32".into()))?,
            u32::try_from(cfg.rope_head_dim)
                .map_err(|_| Error::Model("DeepSeek-V4 DSpark RoPE dim exceeds u32".into()))?,
            false,
        )?;
        operators
            .cuda_mut()?
            .ops
            .fp8_attention_kv_qat_quantize_buffer_in_place(
                &mut arena.kv,
                cfg.head_dim,
                cfg.rope_head_dim,
            )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::KvNorm,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.dspark_hybrid_attention_device_into(
            stage,
            self.layer,
            cfg,
            sequence_tokens,
            &arena.query,
            &arena.kv,
            &mut arena.context,
            dspark,
        )?;
        record_attention_stage(
            operators,
            self.layer,
            DeepSeekV4AttentionProfileStage::SparseAttention,
            stage_start,
        )?;

        operators.cuda_mut()?.rope_tail_rows_from_device(
            &rope_name,
            &mut arena.context,
            u32::try_from(sequence_tokens).map_err(|_| {
                Error::Model("DeepSeek-V4 DSpark proposal position exceeds u32".into())
            })?,
            u32::try_from(rows)
                .map_err(|_| Error::Model("DeepSeek-V4 DSpark row count exceeds u32".into()))?,
            u32::try_from(cfg.num_heads)
                .map_err(|_| Error::Model("DeepSeek-V4 DSpark head count exceeds u32".into()))?,
            u32::try_from(cfg.head_dim)
                .map_err(|_| Error::Model("DeepSeek-V4 DSpark head dim exceeds u32".into()))?,
            u32::try_from(cfg.rope_head_dim)
                .map_err(|_| Error::Model("DeepSeek-V4 DSpark RoPE dim exceeds u32".into()))?,
            true,
        )?;
        operators.cuda_mut()?.mla_output_rows_from_device_into(
            &arena.context,
            rows,
            cfg,
            self.layer,
            &mut arena.latent,
            &mut arena.linear_workspace,
            &mut arena.output,
        )
    }
}

fn record_attention_stage(
    operators: &mut DeepSeekV4OperatorContext,
    layer: usize,
    stage: DeepSeekV4AttentionProfileStage,
    start: Option<Instant>,
) -> Result<()> {
    let Some(elapsed_us) = operators.finish_profile_stage(start).map_err(|error| {
        Error::Internal(format!(
            "DeepSeek-V4 layer {layer} attention stage {stage:?} failed while synchronizing: {error}"
        ))
    })?
    else {
        return Ok(());
    };
    operators.record_attention_stage(layer, stage, elapsed_us);
    Ok(())
}

#[cfg(feature = "cuda")]
fn sync_attention_profile_checkpoint(
    operators: &mut DeepSeekV4OperatorContext,
    layer: usize,
    checkpoint: &str,
) -> Result<()> {
    operators.sync_profile_stream().map_err(|error| {
        Error::Internal(format!(
            "DeepSeek-V4 layer {layer} attention checkpoint {checkpoint} failed while synchronizing: {error}"
        ))
    })
}

#[cfg(feature = "cuda")]
fn required_rope_positions(
    start_position: usize,
    position_stride: usize,
    rows: usize,
) -> Result<usize> {
    if rows == 0 {
        return Err(Error::Model(
            "DeepSeek-V4 CUDA RoPE launch requires at least one row".into(),
        ));
    }
    for (field, value) in [
        ("start_position", start_position),
        ("position_stride", position_stride),
        ("rows", rows),
    ] {
        u32::try_from(value).map_err(|_| {
            Error::Model(format!(
                "DeepSeek-V4 CUDA RoPE {field} exceeds the u32 kernel ABI: {value}"
            ))
        })?;
    }
    let last_offset = (rows - 1)
        .checked_mul(position_stride)
        .ok_or_else(|| Error::Model("DeepSeek-V4 CUDA RoPE row-stride overflow".into()))?;
    start_position
        .checked_add(last_offset)
        .and_then(|position| position.checked_add(1))
        .ok_or_else(|| Error::Model("DeepSeek-V4 CUDA RoPE position overflow".into()))
}

#[cfg(feature = "cuda")]
fn dsv4_fused_indexer_topk_supported(
    operators: &DeepSeekV4OperatorContext,
    cfg: DeepSeekV4AttentionConfig,
    rope_dim: usize,
) -> bool {
    operators.fused_indexer_prefill_topk_enabled()
        && dsv4_fused_indexer_topk_shape_supported(cfg, rope_dim)
}

#[cfg(feature = "cuda")]
fn dsv4_fused_indexer_decode_topk_supported(
    operators: &DeepSeekV4OperatorContext,
    cfg: DeepSeekV4AttentionConfig,
    rope_dim: usize,
) -> bool {
    operators.fused_indexer_decode_topk_enabled()
        && dsv4_fused_indexer_topk_shape_supported(cfg, rope_dim)
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

    /// Fork continuation metadata for a runtime-shared paged prefix.
    ///
    /// Historical window/compressed/indexer values are deliberately absent: the
    /// runtime page table is authoritative and the next packed batch rebuilds its
    /// paged binding.
    pub(crate) fn fork_paged_prefix_metadata(&self) -> Self {
        Self {
            window: DeepSeekV4WindowKvCache::new(self.window.window_size, self.window.head_dim),
            compressed: Vec::new(),
            indexer_compressed: Vec::new(),
            main_compressor: self.main_compressor.clone(),
            indexer_compressor: self.indexer_compressor.clone(),
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

struct DeepSeekV4CompressorProjection {
    kv: Vec<f32>,
    score: Vec<f32>,
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
        let projection = DeepSeekV4CompressorProjection {
            kv: operators.linear_rows(&payload.wkv, hidden, tokens)?,
            score: operators.linear_rows(&payload.wgate, hidden, tokens)?,
        };
        self.prefill_start_projected(payload, tokens, projection, rope_dim, rope, operators)
    }

    #[cfg(feature = "cuda")]
    fn prefill_start_device_output_into(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        layer: usize,
        compressor: DeepSeekV4CudaCompressor,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        rope_name: &str,
        arena: &mut DeepSeekV4CompressorDecodeArena,
        linear_workspace: &mut ferrule_cuda::context::CudaArtifactLinearWorkspace,
        recurrent_state: &mut Option<ferrule_cuda::CudaCompressorRecurrentState>,
        recurrent_needs_reset: &mut bool,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<usize> {
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
        let (kv_linear, gate_linear) = match compressor {
            DeepSeekV4CudaCompressor::Main => (
                DeepSeekV4CudaLinear::MainCompressorKv,
                DeepSeekV4CudaLinear::MainCompressorGate,
            ),
            DeepSeekV4CudaCompressor::Indexer => (
                DeepSeekV4CudaLinear::IndexerCompressorKv,
                DeepSeekV4CudaLinear::IndexerCompressorGate,
            ),
        };
        let _ = (kv_linear, gate_linear, linear_workspace);
        operators.cuda_mut()?.compressor_rows_from_device_into(
            layer,
            compressor,
            hidden_dev,
            tokens,
            &mut arena.kv,
            &mut arena.score,
        )?;

        let groups = operators.cuda_mut()?.compressor_recurrent_seed_prefill(
            layer,
            compressor,
            recurrent_state,
            recurrent_needs_reset,
            &arena.kv,
            &arena.score,
            tokens,
            self.ratio,
            self.head_dim,
            self.out_dim,
            self.overlap,
        )?;
        if groups == 0 {
            return Ok(0);
        }

        operators
            .cuda_mut()?
            .compressor_prefill_softmax_from_device_into(
                layer,
                compressor,
                &arena.kv,
                &arena.score,
                groups,
                self.ratio,
                self.head_dim,
                self.out_dim,
                self.overlap,
                &mut arena.compressed,
            )?;
        operators.cuda_mut()?.rms_norm_compressor_rows_device_into(
            layer,
            compressor,
            &arena.compressed,
            groups,
            1e-6,
            &mut arena.normalized,
        )?;
        let effective_rope_dim = rope_dim.min(self.head_dim);
        let rope_positions = required_rope_positions(0, self.ratio, groups)?;
        operators.cuda_mut()?.ensure_rope_tables_with_params(
            rope_name,
            effective_rope_dim,
            rope,
            rope_positions,
        )?;
        operators.cuda_mut()?.rope_tail_rows_strided_from_device(
            rope_name,
            &mut arena.normalized,
            0,
            self.ratio as u32,
            groups as u32,
            1,
            self.head_dim as u32,
            effective_rope_dim as u32,
            false,
        )?;
        if payload.rotate_for_indexer {
            operators
                .cuda_mut()?
                .ops
                .fp4_hadamard_qat_quantize_buffer_in_place(&mut arena.normalized, self.head_dim)?;
        } else {
            operators
                .cuda_mut()?
                .ops
                .fp8_attention_kv_qat_quantize_buffer_in_place(
                    &mut arena.normalized,
                    self.head_dim,
                    effective_rope_dim,
                )?;
        }
        Ok(groups)
    }

    fn prefill_start_projected(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        tokens: usize,
        projection: DeepSeekV4CompressorProjection,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        let DeepSeekV4CompressorProjection {
            kv: kv_rows,
            score: score_rows,
        } = projection;
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
        let projection = DeepSeekV4CompressorProjection {
            kv: operators.linear_matvec(&payload.wkv, hidden)?,
            score: operators.linear_matvec(&payload.wgate, hidden)?,
        };
        self.append_projected_step(payload, projection, position, rope_dim, rope, operators)
    }

    #[cfg(feature = "cuda")]
    fn append_step_from_device_into(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        layer: usize,
        compressor: DeepSeekV4CudaCompressor,
        hidden_dev: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        rope_name: &str,
        recurrent_state: &mut Option<ferrule_cuda::CudaCompressorRecurrentState>,
        recurrent_needs_reset: &mut bool,
        operators: &mut DeepSeekV4OperatorContext,
        scratch: &mut DeepSeekV4CompressorDecodeArena,
    ) -> Result<bool> {
        self.validate_append_payload(payload)?;
        let (kv_linear, gate_linear) = match compressor {
            DeepSeekV4CudaCompressor::Main => (
                DeepSeekV4CudaLinear::MainCompressorKv,
                DeepSeekV4CudaLinear::MainCompressorGate,
            ),
            DeepSeekV4CudaCompressor::Indexer => (
                DeepSeekV4CudaLinear::IndexerCompressorKv,
                DeepSeekV4CudaLinear::IndexerCompressorGate,
            ),
        };
        operators
            .cuda_mut()?
            .linear_pair_matvec_readonly_from_device_into(
                layer,
                kv_linear,
                gate_linear,
                hidden_dev,
                &mut scratch.kv,
                &mut scratch.score,
            )?;
        self.finish_projected_step_from_device_into(
            layer,
            compressor,
            position,
            rope_dim,
            rope,
            rope_name,
            recurrent_state,
            recurrent_needs_reset,
            operators,
            scratch,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn append_projected_step_from_device_into(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        layer: usize,
        compressor: DeepSeekV4CudaCompressor,
        projected_kv: &ferrule_cuda::context::CudaF32Buffer,
        projected_score: &ferrule_cuda::context::CudaF32Buffer,
        row: usize,
        position: usize,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        rope_name: &str,
        recurrent_state: &mut Option<ferrule_cuda::CudaCompressorRecurrentState>,
        recurrent_needs_reset: &mut bool,
        operators: &mut DeepSeekV4OperatorContext,
        scratch: &mut DeepSeekV4CompressorDecodeArena,
    ) -> Result<bool> {
        self.validate_append_payload(payload)?;
        let offset = row
            .checked_mul(self.out_dim)
            .ok_or_else(|| Error::Model("compressor projected row offset overflow".into()))?;
        if projected_kv.len() < offset + self.out_dim
            || projected_score.len() < offset + self.out_dim
        {
            return Err(Error::Model(
                "compressor projected packed rows are too short".into(),
            ));
        }
        operators.cuda_mut()?.ops.copy_f32_range(
            projected_kv,
            offset,
            &mut scratch.kv,
            0,
            self.out_dim,
        )?;
        operators.cuda_mut()?.ops.copy_f32_range(
            projected_score,
            offset,
            &mut scratch.score,
            0,
            self.out_dim,
        )?;
        self.finish_projected_step_from_device_into(
            layer,
            compressor,
            position,
            rope_dim,
            rope,
            rope_name,
            recurrent_state,
            recurrent_needs_reset,
            operators,
            scratch,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn finish_projected_step_from_device_into(
        &mut self,
        layer: usize,
        compressor: DeepSeekV4CudaCompressor,
        position: usize,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        rope_name: &str,
        recurrent_state: &mut Option<ferrule_cuda::CudaCompressorRecurrentState>,
        recurrent_needs_reset: &mut bool,
        operators: &mut DeepSeekV4OperatorContext,
        scratch: &mut DeepSeekV4CompressorDecodeArena,
    ) -> Result<bool> {
        let boundary = operators.cuda_mut()?.compressor_recurrent_append_into(
            layer,
            compressor,
            recurrent_state,
            recurrent_needs_reset,
            &scratch.kv,
            &scratch.score,
            position,
            self.ratio,
            self.head_dim,
            self.out_dim,
            self.overlap,
            &mut scratch.compressed,
        )?;
        if !boundary {
            return Ok(false);
        }
        operators.cuda_mut()?.rms_norm_compressor_rows_device_into(
            layer,
            compressor,
            &scratch.compressed,
            1,
            1e-6,
            &mut scratch.normalized,
        )?;
        let compressed_position = position + 1 - self.ratio;
        let effective_rope_dim = rope_dim.min(self.head_dim);
        let rope_positions = required_rope_positions(compressed_position, self.ratio, 1)?;
        operators.cuda_mut()?.ensure_rope_tables_with_params(
            rope_name,
            effective_rope_dim,
            rope,
            rope_positions,
        )?;
        operators.cuda_mut()?.rope_tail_rows_strided_from_device(
            rope_name,
            &mut scratch.normalized,
            compressed_position as u32,
            self.ratio as u32,
            1,
            1,
            self.head_dim as u32,
            effective_rope_dim as u32,
            false,
        )?;
        if compressor == DeepSeekV4CudaCompressor::Indexer {
            operators
                .cuda_mut()?
                .ops
                .fp4_hadamard_qat_quantize_buffer_in_place(
                    &mut scratch.normalized,
                    self.head_dim,
                )?;
        } else {
            operators
                .cuda_mut()?
                .ops
                .fp8_attention_kv_qat_quantize_buffer_in_place(
                    &mut scratch.normalized,
                    self.head_dim,
                    effective_rope_dim,
                )?;
        }
        Ok(true)
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

    fn append_projected_state(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        kv: Vec<f32>,
        mut score: Vec<f32>,
        position: usize,
    ) -> Result<bool> {
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
        Ok((position + 1).is_multiple_of(self.ratio))
    }

    fn advance_overlap_state(&mut self) {
        if self.overlap {
            let split = self.ratio * self.out_dim;
            let (prev, current) = self.kv_state.split_at_mut(split);
            prev.copy_from_slice(current);
            let (prev, current) = self.score_state.split_at_mut(split);
            prev.copy_from_slice(current);
        }
    }

    fn append_projected_step(
        &mut self,
        payload: &DeepSeekV4CompressorPayload,
        projection: DeepSeekV4CompressorProjection,
        position: usize,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Option<Vec<f32>>> {
        if !self.append_projected_state(payload, projection.kv, projection.score, position)? {
            return Ok(None);
        }

        let compressed = self.compress_current_window(
            payload,
            position + 1 - self.ratio,
            rope_dim,
            rope,
            operators,
        )?;
        self.advance_overlap_state();
        Ok(Some(compressed))
    }

    fn current_window_rows(&self) -> (usize, Vec<f32>, Vec<f32>) {
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
        (rows, kv_rows, score_rows)
    }

    fn compress_current_window(
        &self,
        payload: &DeepSeekV4CompressorPayload,
        compressed_position: usize,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        let (rows, kv_rows, score_rows) = self.current_window_rows();
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

pub struct DeepSeekV4WindowKvCache {
    window_size: usize,
    pub(crate) head_dim: usize,
    len: usize,
    values: Vec<f32>,
    #[cfg(feature = "cuda")]
    cuda: DeepSeekV4CudaSequenceKvState,
}

impl Clone for DeepSeekV4WindowKvCache {
    fn clone(&self) -> Self {
        Self {
            window_size: self.window_size,
            head_dim: self.head_dim,
            len: self.len,
            values: self.values.clone(),
            #[cfg(feature = "cuda")]
            cuda: DeepSeekV4CudaSequenceKvState::default(),
        }
    }
}

impl PartialEq for DeepSeekV4WindowKvCache {
    fn eq(&self, other: &Self) -> bool {
        self.window_size == other.window_size
            && self.head_dim == other.head_dim
            && self.len == other.len
            && self.values == other.values
    }
}

impl std::fmt::Debug for DeepSeekV4WindowKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("DeepSeekV4WindowKvCache");
        debug
            .field("window_size", &self.window_size)
            .field("head_dim", &self.head_dim)
            .field("len", &self.len)
            .field("values", &self.values);
        #[cfg(feature = "cuda")]
        debug.field("cuda", &self.cuda);
        debug.finish()
    }
}

impl DeepSeekV4WindowKvCache {
    pub fn new(window_size: usize, head_dim: usize) -> Self {
        Self {
            window_size,
            head_dim,
            len: 0,
            values: vec![0.0; window_size * head_dim],
            #[cfg(feature = "cuda")]
            cuda: DeepSeekV4CudaSequenceKvState::default(),
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
        #[cfg(feature = "cuda")]
        self.cuda.reset_for_reuse();
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
        self.record_device_append();
        Ok(())
    }

    /// Advance only the ring-validity metadata after successful device KV
    /// writes. CUDA paths intentionally do not mirror values back to host, but
    /// graph preparation and top-k masking still require the same valid length.
    fn record_device_append(&mut self) {
        self.record_device_rows(1);
    }

    pub(crate) fn record_device_rows(&mut self, rows: usize) {
        self.len = self.window_size.min(self.len.saturating_add(rows));
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

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_state_mut(&mut self) -> &mut DeepSeekV4CudaSequenceKvState {
        &mut self.cuda
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fork_test_config() -> DeepSeekV4AttentionConfig {
        DeepSeekV4AttentionConfig {
            hidden_size: 4,
            num_heads: 1,
            head_dim: 2,
            q_lora_rank: 2,
            rope_head_dim: 0,
            o_groups: 1,
            o_lora_rank: 2,
            window_size: 4,
            compress_ratio: 2,
            norm_eps: 1e-6,
            rope_theta: 10_000.0,
            compress_rope_theta: 10_000.0,
            original_seq_len: 16,
            rope_factor: 1.0,
            beta_fast: 32,
            beta_slow: 1,
            index_n_heads: 1,
            index_head_dim: 2,
            index_topk: 1,
        }
    }

    #[test]
    fn paged_prefix_fork_copies_recurrent_metadata_without_history() -> Result<()> {
        let mut source = DeepSeekV4AttentionCache::new(fork_test_config());
        source.window.append(0, &[1.0, 2.0])?;
        source.compressed.extend_from_slice(&[3.0, 4.0]);
        source.indexer_compressed.extend_from_slice(&[5.0, 6.0]);
        source.main_compressor.as_mut().unwrap().kv_state[0] = 7.0;

        let fork = source.fork_paged_prefix_metadata();
        assert!(fork.window.is_empty());
        assert!(fork.window.values_full().iter().all(|value| *value == 0.0));
        assert!(fork.compressed.is_empty());
        assert!(fork.indexer_compressed.is_empty());
        assert_eq!(fork.main_compressor, source.main_compressor);
        assert_eq!(
            fork.main_compressor.as_ref().unwrap().kv_state[0],
            7.0,
            "compressor recurrent continuation must be retained"
        );

        Ok(())
    }

    #[test]
    fn window_kv_clone_copies_host_semantics_independently() -> Result<()> {
        let mut original = DeepSeekV4WindowKvCache::new(2, 2);
        original.append(0, &[1.0, 2.0])?;

        let mut cloned = original.clone();
        assert_eq!(cloned, original);
        #[cfg(feature = "cuda")]
        assert_eq!(cloned, original, "CUDA metadata is excluded from equality");

        cloned.append(1, &[3.0, 4.0])?;
        assert_ne!(cloned, original);
        assert_eq!(original.len(), 1);
        assert_eq!(original.values_full(), &[1.0, 2.0, 0.0, 0.0]);
        Ok(())
    }
}
