//! CUDA MLA/HyperConnection/RoutedMoE transformer components.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ferrule_backend::cuda::operators::{
    attention as cuda_attention, linear as cuda_linear, moe as cuda_moe,
};
use ferrule_backend::plan::{
    ExecutionMode, KernelOperation, LaunchDescriptor, LayerKernelRequirements,
    LinearBundleRequirement, ModelKernelPlan, OperationRequirement, WeightLayout,
};

use ferrule_common::execution::ForwardPhase;
use ferrule_common::{CompletionHub, Error, ResidencyLeaseSet, Result};

use crate::checkpoint::weight::{LinearWeight, LinearWeightFormat};
use crate::checkpoint::{CheckpointDType, CheckpointMatrixSlice};
use crate::moe::cuda_materialization::CudaSharedExpertSubsystem;
use crate::moe::streaming::ExpertSourceCatalog;
use crate::moe::{
    PreparedRoutedMoe, RoutedMoeAttribution, RoutedMoeCancelProgress, RoutedMoeContinuation,
    RoutedMoeExecution, RoutedMoePayload, RoutedMoeProgress, RoutedMoeScratch,
    RoutedMoeSequenceEvent,
};
use crate::transformer::attention::mla::{
    MlaCompressor, MlaDecodeArena, MlaExecution, MlaKvView, MlaLayerState, MlaPagedKvBinding,
    MlaProposalAttentionBuffers, MlaProposalMainBuffers, MlaRowsTransitionArena, PreparedMla,
    PreparedMlaCompressor, PreparedMlaIndexer, PreparedMlaOutput, PreparedMlaWeights,
};
use crate::transformer::connection::{
    HyperConnection, HyperConnectionHead, HyperConnectionPostBuffers, HyperConnectionPreBuffers,
    HyperConnectionStage, PreparedHyperConnection, PreparedHyperConnectionHead,
};
use crate::transformer::proposal::{
    MtpAttachment, PreparedMtpAttachment, PreparedMtpHeads, PreparedMtpStage,
};
use crate::transformer::{
    Connected, HyperReduction, MtpTap, OutputPipeline, PreparedCudaLinear,
    PreparedDecoderAttachment, PreparedTransformer, TransformerLayer,
};

use crate::execution::ExecutionShapeKey;

#[cfg(feature = "cuda")]
pub(crate) struct CudaTransformerBuffers {
    pub(crate) hc_input: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(crate) final_hidden: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(crate) topk_row: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
}

#[cfg(feature = "cuda")]
fn debug_cuda_stage(
    layer: usize,
    position: Option<usize>,
    stage: &str,
    buffer: &ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    operators: &mut CudaTransformerRuntime,
) -> Result<()> {
    if std::env::var("FERRULE_DEBUG_STAGE_LAYER")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        != Some(layer)
        || std::env::var("FERRULE_DEBUG_STAGE_POSITION")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .is_some_and(|expected| position != Some(expected))
    {
        return Ok(());
    }
    let values = operators.ops.download_f32_buffer(buffer)?;
    if let Some(directory) = std::env::var_os("FERRULE_DEBUG_STAGE_DUMP_DIR") {
        std::fs::create_dir_all(&directory).map_err(|source| Error::Internal {
            message: format!("failed to create stage dump directory: {source}"),
        })?;
        let path = std::path::PathBuf::from(directory).join(format!("layer_{layer}_{stage}.f32"));
        if !path.exists() {
            let bytes = values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>();
            std::fs::write(&path, bytes).map_err(|source| Error::Internal {
                message: format!("failed to write stage dump {}: {source}", path.display()),
            })?;
        }
    }
    let sum = values.iter().copied().sum::<f32>();
    let sumsq = values.iter().map(|value| value * value).sum::<f32>();
    let absmax = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "stage layer={} name={} sum={} sumsq={} absmax={} samples={:?}",
        layer,
        stage,
        sum,
        sumsq,
        absmax,
        &values[..values.len().min(4)]
    );
    Ok(())
}

pub struct MlaHyperMoeLayer {
    pub layer: usize,
    pub hyper_connection: HyperConnection,
    pub attention: PreparedMla,
    pub feed_forward: RoutedMoePayload,
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaMlaHyperMoeContinuation {
    rows: usize,
    position: Option<usize>,
    moe: RoutedMoeContinuation,
}

#[cfg(feature = "cuda")]
impl CudaMlaHyperMoeContinuation {
    pub(crate) fn pending_experts(&self) -> Vec<crate::moe::RoutedMoePendingExpert> {
        self.moe.pending_experts()
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::large_enum_variant)]
pub(crate) enum CudaMlaHyperMoeProgress {
    Waiting(CudaMlaHyperMoeContinuation),
    Complete { events: Vec<RoutedMoeSequenceEvent> },
}

/// Exact per-row dimensions of one compressor's CUDA scratch buffers.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MlaCompressorArenaShapeKey {
    compress_ratio: usize,
    head_dim: usize,
    overlap: bool,
    rotate_for_indexer: bool,
    ape_rows: usize,
    ape_cols: usize,
    norm_len: usize,
    kv_width: usize,
    score_width: usize,
    compressed_width: usize,
    normalized_width: usize,
}

/// Exact per-row dimensions of every CUDA attention arena buffer.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PreparedMlaArenaShapeKey {
    q_latent_width: usize,
    q_norm_width: usize,
    q_indexer_width: usize,
    query_raw_width: usize,
    query_width: usize,
    kv_raw_width: usize,
    kv_width: usize,
    index_query_width: usize,
    index_weights_width: usize,
    topk_width: usize,
    empty_query_width: usize,
    empty_weights_width: usize,
    empty_kv_width: usize,
    context_width: usize,
    latent_width: usize,
    output_width: usize,
    linear_workspace_width: usize,
    main_compressor: Option<MlaCompressorArenaShapeKey>,
    indexer_compressor: Option<MlaCompressorArenaShapeKey>,
    indexer_query_b_input_width: Option<usize>,
    indexer_query_b_output_width: Option<usize>,
    indexer_weights_input_width: Option<usize>,
    indexer_weights_output_width: Option<usize>,
}

/// Shape key for all buffers owned by one layer scratch arena.
///
/// `rows` and execution phase remain in the outer `ExecutionShapeKey` bucket;
/// this key distinguishes the exact arena variants within that bucket.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CudaMlaHyperMoeArenaShapeKey {
    attn_hidden_width: usize,
    attn_pre_width: usize,
    attn_post_width: usize,
    attn_comb_width: usize,
    attn_norm_width: usize,
    after_attn_width: usize,
    ffn_hidden_width: usize,
    ffn_pre_width: usize,
    ffn_post_width: usize,
    ffn_comb_width: usize,
    ffn_norm_width: usize,
    attention: PreparedMlaArenaShapeKey,
    router_logits_width: usize,
    router_indices_width: usize,
    router_weights_width: usize,
    moe_output_width: usize,
    shared_gate_input_width: usize,
    shared_gate_output_width: usize,
    shared_up_input_width: usize,
    shared_up_output_width: usize,
    shared_down_input_width: usize,
    shared_down_output_width: usize,
    moe_route_count: usize,
    moe_route_output_width: usize,
    layer_output_width: usize,
}

#[cfg(any(feature = "cuda", test))]
impl MlaCompressorArenaShapeKey {
    fn from_payload(payload: &crate::transformer::attention::mla::MlaCompressor) -> Self {
        Self {
            compress_ratio: payload.compress_ratio,
            head_dim: payload.head_dim,
            overlap: payload.overlap,
            rotate_for_indexer: payload.rotate_for_indexer,
            ape_rows: payload.ape_rows,
            ape_cols: payload.ape_cols,
            norm_len: payload.norm.len(),
            kv_width: payload.wkv.format.out_features(),
            score_width: payload.wgate.format.out_features(),
            compressed_width: payload.head_dim,
            normalized_width: payload.head_dim,
        }
    }
}

#[cfg(any(feature = "cuda", test))]
impl CudaMlaHyperMoeArenaShapeKey {
    fn for_layer(layer: &MlaHyperMoeLayer) -> Self {
        let config = layer.hyper_connection.config();
        let hc = config.hc_mult;
        let hidden = config.hidden_size;
        let hc_dim = hc * hidden;
        let hc_comb = hc * hc;
        let cfg = layer.attention.config();
        let q_full = cfg.q_full_dim();
        let output_latent = cfg.output_latent_dim();
        let index_query = cfg.index_n_heads * cfg.index_head_dim;
        let compressed = layer.attention.compressed.as_ref();
        let main_compressor =
            compressed.map(|payload| MlaCompressorArenaShapeKey::from_payload(&payload.compressor));
        let indexer = compressed.and_then(|payload| payload.indexer.as_ref());
        let indexer_compressor =
            indexer.map(|payload| MlaCompressorArenaShapeKey::from_payload(&payload.compressor));
        let linear_workspace_width = cfg
            .hidden_size
            .max(cfg.q_lora_rank)
            .max(q_full)
            .max(cfg.head_dim)
            .max(output_latent)
            .max(index_query);

        Self {
            attn_hidden_width: hidden,
            attn_pre_width: hc,
            attn_post_width: hc,
            attn_comb_width: hc_comb,
            attn_norm_width: hidden,
            after_attn_width: hc_dim,
            ffn_hidden_width: hidden,
            ffn_pre_width: hc,
            ffn_post_width: hc,
            ffn_comb_width: hc_comb,
            ffn_norm_width: hidden,
            attention: PreparedMlaArenaShapeKey {
                q_latent_width: cfg.q_lora_rank,
                q_norm_width: cfg.q_lora_rank,
                q_indexer_width: cfg.q_lora_rank,
                query_raw_width: q_full,
                query_width: q_full,
                kv_raw_width: cfg.head_dim,
                kv_width: cfg.head_dim,
                index_query_width: index_query,
                index_weights_width: cfg.index_n_heads,
                topk_width: cfg.window_size + cfg.index_topk,
                empty_query_width: 1,
                empty_weights_width: 1,
                empty_kv_width: 1,
                context_width: q_full,
                latent_width: output_latent,
                output_width: cfg.hidden_size,
                linear_workspace_width,
                main_compressor,
                indexer_compressor,
                indexer_query_b_input_width: indexer
                    .map(|payload| payload.wq_b.format.in_features()),
                indexer_query_b_output_width: indexer
                    .map(|payload| payload.wq_b.format.out_features()),
                indexer_weights_input_width: indexer
                    .map(|payload| payload.weights_proj.format.in_features()),
                indexer_weights_output_width: indexer
                    .map(|payload| payload.weights_proj.format.out_features()),
            },
            router_logits_width: layer.feed_forward.expert_count(),
            router_indices_width: layer.feed_forward.router_policy.top_k,
            router_weights_width: layer.feed_forward.router_policy.top_k,
            moe_output_width: hidden,
            shared_gate_input_width: layer.feed_forward.shared_expert.gate.format.in_features(),
            shared_gate_output_width: layer.feed_forward.shared_expert.gate.format.out_features(),
            shared_up_input_width: layer.feed_forward.shared_expert.up.format.in_features(),
            shared_up_output_width: layer.feed_forward.shared_expert.up.format.out_features(),
            shared_down_input_width: layer.feed_forward.shared_expert.down.format.in_features(),
            shared_down_output_width: layer.feed_forward.shared_expert.down.format.out_features(),
            moe_route_count: layer.feed_forward.router_policy.top_k,
            moe_route_output_width: hidden,
            layer_output_width: hc_dim,
        }
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn layer_arena_variant_layout(layers: &[MlaHyperMoeLayer]) -> (Vec<usize>, Vec<usize>) {
    let mut variants = HashMap::new();
    let mut layer_to_variant = Vec::with_capacity(layers.len());
    let mut representative_layers = Vec::new();
    for (layer_idx, layer) in layers.iter().enumerate() {
        let key = CudaMlaHyperMoeArenaShapeKey::for_layer(layer);
        let variant = match variants.get(&key).copied() {
            Some(variant) => variant,
            None => {
                let variant = representative_layers.len();
                variants.insert(key, variant);
                representative_layers.push(layer_idx);
                variant
            }
        };
        layer_to_variant.push(variant);
    }
    (layer_to_variant, representative_layers)
}

pub(crate) enum CudaMlaLayerRequest<'a> {
    Target {
        kv: &'a mut MlaKvView,
        states: &'a mut [&'a mut MlaLayerState],
        row_to_sequence: &'a [usize],
        sequence_major_rows: &'a [usize],
        sequence_phases: &'a [ForwardPhase],
        paged_bindings: &'a [MlaPagedKvBinding],
        token_ids: &'a [u32],
        positions: &'a [usize],
    },
    Proposal {
        kv: &'a mut MlaKvView,
        stage: usize,
        sequence_tokens: usize,
        token_ids: &'a [u32],
        attention: &'a mut MlaProposalAttentionBuffers,
    },
}

impl MlaHyperMoeLayer {
    /// Starts the shared MLA/HyperConnection/RoutedMoE layer composition.
    /// Target and proposal modes differ only in the MLA attention semantic.
    #[cfg(feature = "cuda")]
    pub(crate) fn begin_device_hc_device(
        &self,
        mut request: CudaMlaLayerRequest<'_>,
        arena: &mut CudaMlaHyperMoeArena,
        hc_state_dev: &mut ferrule_backend::cuda::operators::linear::CudaF32Buffer,
        operators: &mut CudaTransformerRuntime,
    ) -> Result<CudaMlaHyperMoeProgress> {
        let (rows, position) = match &request {
            CudaMlaLayerRequest::Target {
                states,
                row_to_sequence,
                sequence_major_rows,
                sequence_phases,
                paged_bindings,
                token_ids,
                positions,
                ..
            } => {
                let rows = token_ids.len();
                if rows == 0
                    || positions.len() != rows
                    || row_to_sequence.len() != rows
                    || sequence_major_rows.len() != rows
                    || states.len() != paged_bindings.len()
                    || sequence_phases.len() != states.len()
                    || row_to_sequence
                        .iter()
                        .any(|sequence| *sequence >= states.len())
                {
                    return Err(Error::Model {
                        message: "CUDA transformer packed row/sequence metadata is inconsistent"
                            .into(),
                    });
                }
                (rows, (rows == 1).then_some(positions[0]))
            }
            CudaMlaLayerRequest::Proposal {
                stage,
                sequence_tokens,
                token_ids,
                ..
            } => {
                let rows = ferrule_backend::cuda::operators::linear::PROPOSAL_ROWS;
                if token_ids.len() != rows || *sequence_tokens == 0 {
                    return Err(Error::Model {
                        message: format!(
                            "CUDA transformer proposal stage {stage} input mismatch: tokens={} sequence_tokens={sequence_tokens} expected_tokens={rows}",
                            token_ids.len(),
                        ),
                    });
                }
                (rows, None)
            }
        };
        let hyper_connection_config = self.hyper_connection.config();
        if hc_state_dev.len() != rows.saturating_mul(hyper_connection_config.hc_hidden_size()) {
            return Err(Error::Model {
                message: format!(
                    "CUDA transformer layer {} HC input mismatch: got {} expected {}",
                    self.layer,
                    hc_state_dev.len(),
                    rows.saturating_mul(hyper_connection_config.hc_hidden_size()),
                ),
            });
        }

        operators.require_inference_operation(
            self.layer,
            HyperConnectionStage::Attention.kernel_operation(),
        )?;
        let attention_fp8 = operators
            .prepared_layer(self.layer)?
            .attention()
            .connection()
            .pre(
                &operators.ops,
                HyperConnectionStage::Attention,
                hc_state_dev,
                rows,
                HyperConnectionPreBuffers {
                    hidden: &mut arena.attn_hidden,
                    normalized: &mut arena.attn_norm,
                    mix: &mut arena.hc_mix,
                    workspace: &mut arena.hc_workspace,
                    split_pre: &mut arena.attn_pre,
                    split_post: &mut arena.attn_post,
                    split_comb: &mut arena.attn_comb,
                    packed: &mut arena.hc_fp8_pack,
                },
            )?;
        debug_cuda_stage(
            self.layer,
            position,
            "attn_hc_pre",
            &arena.attn_hidden,
            operators,
        )?;
        debug_cuda_stage(
            self.layer,
            position,
            "attn_hc_post_weights",
            &arena.attn_post,
            operators,
        )?;
        debug_cuda_stage(
            self.layer,
            position,
            "attn_hc_comb",
            &arena.attn_comb,
            operators,
        )?;
        debug_cuda_stage(
            self.layer,
            position,
            "attn_norm",
            &arena.attn_norm,
            operators,
        )?;

        match &mut request {
            CudaMlaLayerRequest::Target {
                kv,
                states,
                row_to_sequence,
                sequence_major_rows,
                sequence_phases,
                paged_bindings,
                positions,
                ..
            } => {
                let transition =
                    arena
                        .attention_transition
                        .as_mut()
                        .ok_or_else(|| Error::Internal {
                            message: "packed decode arena is missing attention transition scratch"
                                .into(),
                        })?;
                let mut attention_caches = states
                    .iter_mut()
                    .map(|state| &mut state.kv)
                    .collect::<Vec<_>>();
                self.attention.packed_rows_from_device_into(
                    kv,
                    &mut attention_caches,
                    &arena.attn_norm,
                    &attention_fp8,
                    positions,
                    row_to_sequence,
                    sequence_major_rows,
                    sequence_phases,
                    paged_bindings,
                    &mut operators.mla,
                    &mut arena.attention,
                    transition,
                )?;
            }
            CudaMlaLayerRequest::Proposal {
                kv,
                stage,
                sequence_tokens,
                attention,
                ..
            } => self.attention.proposal_block_from_device_into(
                kv,
                *stage,
                &attention_fp8,
                *sequence_tokens,
                &mut operators.mla,
                &mut arena.attention,
                attention,
            )?,
        }
        debug_cuda_stage(
            self.layer,
            position,
            "attention_output",
            &arena.attention.output,
            operators,
        )?;

        operators
            .prepared_layer(self.layer)?
            .attention()
            .connection()
            .post(
                &operators.ops,
                rows,
                HyperConnectionPostBuffers {
                    hidden: &arena.attention.output,
                    residual: hc_state_dev,
                    split_post: &arena.attn_post,
                    split_comb: &arena.attn_comb,
                    output: &mut arena.after_attn,
                },
            )?;
        debug_cuda_stage(
            self.layer,
            position,
            "attention_hc_post",
            &arena.after_attn,
            operators,
        )?;

        operators.require_inference_operation(
            self.layer,
            HyperConnectionStage::FeedForward.kernel_operation(),
        )?;
        let ffn_fp8 = operators
            .prepared_layer(self.layer)?
            .feed_forward()
            .connection()
            .pre(
                &operators.ops,
                HyperConnectionStage::FeedForward,
                &arena.after_attn,
                rows,
                HyperConnectionPreBuffers {
                    hidden: &mut arena.ffn_hidden,
                    normalized: &mut arena.ffn_norm,
                    mix: &mut arena.hc_mix,
                    workspace: &mut arena.hc_workspace,
                    split_pre: &mut arena.ffn_pre,
                    split_post: &mut arena.ffn_post,
                    split_comb: &mut arena.ffn_comb,
                    packed: &mut arena.hc_fp8_pack,
                },
            )?;
        debug_cuda_stage(
            self.layer,
            position,
            "ffn_hc_pre",
            &arena.ffn_hidden,
            operators,
        )?;
        debug_cuda_stage(
            self.layer,
            position,
            "ffn_hc_post_weights",
            &arena.ffn_post,
            operators,
        )?;
        debug_cuda_stage(
            self.layer,
            position,
            "ffn_hc_comb",
            &arena.ffn_comb,
            operators,
        )?;
        debug_cuda_stage(self.layer, position, "ffn_norm", &arena.ffn_norm, operators)?;

        let (prepared_moe, execution) = operators.routed_moe_parts(self.layer)?;
        let moe = match request {
            CudaMlaLayerRequest::Target {
                row_to_sequence,
                sequence_phases,
                token_ids,
                ..
            } => {
                let attribution_phases = sequence_phases
                    .iter()
                    .map(|phase| match phase {
                        ForwardPhase::Prefill => crate::moe::prediction::ExpertAccessPhase::Prefill,
                        ForwardPhase::Decode => crate::moe::prediction::ExpertAccessPhase::Decode,
                    })
                    .collect::<Vec<_>>();
                prepared_moe.start(
                    execution,
                    &arena.ffn_norm,
                    &ffn_fp8,
                    token_ids,
                    RoutedMoeAttribution::packed(row_to_sequence, &attribution_phases),
                    &mut arena.moe,
                )?
            }
            CudaMlaLayerRequest::Proposal { token_ids, .. } => {
                let row_to_sequence =
                    [0usize; ferrule_backend::cuda::operators::linear::PROPOSAL_ROWS];
                let sequence_phases = [crate::moe::prediction::ExpertAccessPhase::Decode];
                prepared_moe.start(
                    execution,
                    &arena.ffn_norm,
                    &ffn_fp8,
                    token_ids,
                    RoutedMoeAttribution::packed(&row_to_sequence, &sequence_phases),
                    &mut arena.moe,
                )?
            }
        };
        debug_cuda_stage(
            self.layer,
            position,
            "shared_moe_output",
            arena.moe.output(),
            operators,
        )?;
        Ok(CudaMlaHyperMoeProgress::Waiting(
            CudaMlaHyperMoeContinuation {
                rows,
                position,
                moe,
            },
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn resume_device_hc_device(
        &self,
        continuation: CudaMlaHyperMoeContinuation,
        leases: Option<&ResidencyLeaseSet>,
        arena: &mut CudaMlaHyperMoeArena,
        hc_state_dev: &mut ferrule_backend::cuda::operators::linear::CudaF32Buffer,
        operators: &mut CudaTransformerRuntime,
    ) -> Result<CudaMlaHyperMoeProgress> {
        let CudaMlaHyperMoeContinuation {
            rows,
            position,
            moe,
        } = continuation;
        let (prepared_moe, execution) = operators.routed_moe_parts(self.layer)?;
        match prepared_moe.resume(execution, moe, leases, &arena.ffn_norm, &mut arena.moe)? {
            RoutedMoeProgress::Waiting(moe) => Ok(CudaMlaHyperMoeProgress::Waiting(
                CudaMlaHyperMoeContinuation {
                    rows,
                    position,
                    moe,
                },
            )),
            RoutedMoeProgress::Complete { events } => {
                debug_cuda_stage(
                    self.layer,
                    position,
                    "route_output",
                    arena.moe.route_output(),
                    operators,
                )?;
                debug_cuda_stage(
                    self.layer,
                    position,
                    "moe_output",
                    arena.moe.output(),
                    operators,
                )?;
                operators
                    .prepared_layer(self.layer)?
                    .feed_forward()
                    .connection()
                    .post(
                        &operators.ops,
                        rows,
                        HyperConnectionPostBuffers {
                            hidden: arena.moe.output(),
                            residual: &arena.after_attn,
                            split_post: &arena.ffn_post,
                            split_comb: &arena.ffn_comb,
                            output: &mut arena.layer_output,
                        },
                    )?;
                debug_cuda_stage(
                    self.layer,
                    position,
                    "layer_output",
                    &arena.layer_output,
                    operators,
                )?;
                std::mem::swap(hc_state_dev, &mut arena.layer_output);
                Ok(CudaMlaHyperMoeProgress::Complete { events })
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn poll_cancel_ready(
        &self,
        continuation: &mut CudaMlaHyperMoeContinuation,
        operators: &mut CudaTransformerRuntime,
    ) -> Result<bool> {
        let (prepared, execution) = operators.routed_moe_parts(self.layer)?;
        prepared
            .cancel(execution, &mut continuation.moe)
            .map(|progress| matches!(progress, RoutedMoeCancelProgress::Complete))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cancel_device_hc_device(
        &self,
        mut continuation: CudaMlaHyperMoeContinuation,
        operators: &mut CudaTransformerRuntime,
    ) -> Result<()> {
        let (prepared, execution) = operators.routed_moe_parts(self.layer)?;
        match prepared.cancel(execution, &mut continuation.moe)? {
            RoutedMoeCancelProgress::Complete => Ok(()),
            RoutedMoeCancelProgress::Waiting => Err(Error::Execution {
                message: "routed-MoE cancellation was consumed before quiescence".into(),
            }),
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaMlaHyperMoeArena {
    pub(super) attn_hidden: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(super) attn_pre: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(super) attn_post: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(super) attn_comb: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(super) attn_norm: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(super) hc_mix: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(super) hc_workspace: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(super) hc_fp8_pack: ferrule_backend::cuda::operators::linear::CudaFp8ActivationPack,
    pub(super) after_attn: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(super) ffn_hidden: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(super) ffn_pre: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(super) ffn_post: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(super) ffn_comb: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(super) ffn_norm: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    pub(super) attention: MlaDecodeArena,
    attention_transition: Option<MlaRowsTransitionArena>,
    pub(super) moe: RoutedMoeScratch,
    pub(super) layer_output: ferrule_backend::cuda::operators::linear::CudaF32Buffer,
    hidden_size: usize,
    hc_mult: usize,
}

/// One exact phase/rows bucket containing one arena per unique transformer scratch shape.
#[cfg(feature = "cuda")]
pub(crate) struct CudaMlaHyperMoeArenaVariants {
    arenas: Vec<CudaMlaHyperMoeArena>,
    layer_to_variant: Box<[usize]>,
}

#[cfg(feature = "cuda")]
impl CudaMlaHyperMoeArenaVariants {
    pub(crate) fn try_build_for_packed_mode(
        layers: &[MlaHyperMoeLayer],
        rows: usize,
        operators: &mut CudaTransformerRuntime,
    ) -> Result<Self> {
        Self::try_build_with_row_transitions(layers, rows, true, true, operators)
    }

    fn try_build_with_row_transitions(
        layers: &[MlaHyperMoeLayer],
        rows: usize,
        independent_rows: bool,
        allocate_row_transition: bool,
        operators: &mut CudaTransformerRuntime,
    ) -> Result<Self> {
        let (layer_to_variant, representative_layers) = layer_arena_variant_layout(layers);
        let mut arenas = Vec::with_capacity(representative_layers.len());
        for layer_idx in representative_layers {
            arenas.push(CudaMlaHyperMoeArena::new(
                &layers[layer_idx],
                rows,
                independent_rows,
                allocate_row_transition,
                operators,
            )?);
        }

        Ok(Self {
            arenas,
            layer_to_variant: layer_to_variant.into_boxed_slice(),
        })
    }

    pub(crate) fn get_for_layer_mut(&mut self, layer: usize) -> Option<&mut CudaMlaHyperMoeArena> {
        let variant = *self.layer_to_variant.get(layer)?;
        self.arenas.get_mut(variant)
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for CudaMlaHyperMoeArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaMlaHyperMoeArena")
            .field("hidden_size", &self.hidden_size)
            .field("hyper_connection_streams", &self.hc_mult)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
impl CudaMlaHyperMoeArena {
    pub(crate) fn new(
        layer: &MlaHyperMoeLayer,
        rows: usize,
        independent_rows: bool,
        allocate_row_transition: bool,
        operators: &mut CudaTransformerRuntime,
    ) -> Result<Self> {
        let config = layer.hyper_connection.config();
        let hidden = config.hidden_size;
        let hc = config.hc_mult;
        let hc_dim = hc.checked_mul(hidden).ok_or_else(|| Error::Internal {
            message: "CUDA transformer arena HC dim overflow".into(),
        })?;
        let comb = hc.checked_mul(hc).ok_or_else(|| Error::Internal {
            message: "CUDA transformer arena HC comb overflow".into(),
        })?;
        Ok(Self {
            attn_hidden: operators.ops.zero_f32_buffer(rows * hidden)?,
            attn_pre: operators.ops.zero_f32_buffer(rows * hc)?,
            attn_post: operators.ops.zero_f32_buffer(rows * hc)?,
            attn_comb: operators.ops.zero_f32_buffer(rows * comb)?,
            attn_norm: operators.ops.zero_f32_buffer(rows * hidden)?,
            hc_mix: operators.ops.zero_f32_buffer(rows * config.mix_hc())?,
            hc_workspace: operators
                .ops
                .zero_f32_buffer(rows * config.mix_hc() * 64 + 1)?,
            hc_fp8_pack: operators.ops.fp8_activation_pack(rows, hidden)?,
            after_attn: operators.ops.zero_f32_buffer(rows * hc_dim)?,
            ffn_hidden: operators.ops.zero_f32_buffer(rows * hidden)?,
            ffn_pre: operators.ops.zero_f32_buffer(rows * hc)?,
            ffn_post: operators.ops.zero_f32_buffer(rows * hc)?,
            ffn_comb: operators.ops.zero_f32_buffer(rows * comb)?,
            ffn_norm: operators.ops.zero_f32_buffer(rows * hidden)?,
            attention: MlaDecodeArena::new(
                &layer.attention,
                rows,
                independent_rows,
                &mut operators.mla,
            )?,
            attention_transition: allocate_row_transition
                .then(|| MlaRowsTransitionArena::new(&layer.attention, &mut operators.mla))
                .transpose()?,
            moe: RoutedMoeScratch::new(
                operators
                    .prepared_layer(layer.layer)?
                    .feed_forward()
                    .block(),
                rows,
                &operators.ops,
            )?,
            layer_output: operators.ops.zero_f32_buffer(rows * hc_dim)?,
            hidden_size: hidden,
            hc_mult: hc,
        })
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaTransformerRowLayout {
    IndependentRows,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaTransformerArenaKey {
    shape: ExecutionShapeKey,
    row_layout: CudaTransformerRowLayout,
}

#[cfg(feature = "cuda")]
impl CudaTransformerArenaKey {
    pub const fn new(shape: ExecutionShapeKey, row_layout: CudaTransformerRowLayout) -> Self {
        Self { shape, row_layout }
    }

    pub const fn shape(&self) -> ExecutionShapeKey {
        self.shape
    }
}

#[cfg(feature = "cuda")]
pub struct CudaMtpHeadBuffers {
    pub(crate) workspace: ferrule_backend::cuda::operators::attention::CudaProposalHeadWorkspace,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct CudaTransformerMetrics {
    output_head_calls: u64,
    output_head_rows: u64,
    output_head_topk_us: u64,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CudaTransformerRuntimeConfig {
    profile: bool,
    profile_sync: bool,
    managed_experts: bool,
}

#[cfg(feature = "cuda")]
impl CudaTransformerRuntimeConfig {
    pub(crate) const fn new(profile: bool, profile_sync: bool, managed_experts: bool) -> Self {
        Self {
            profile,
            profile_sync,
            managed_experts,
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTransformerMtpSource<'a, E> {
    pub(crate) attachment: &'a MtpAttachment<MlaHyperMoeLayer, E>,
    pub(crate) kernel_plan: &'a ModelKernelPlan,
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTransformerSource<'a, E> {
    pub(crate) embedding: &'a CheckpointMatrixSlice,
    pub(crate) output_norm: &'a [f32],
    pub(crate) output_head: &'a CheckpointMatrixSlice,
    pub(crate) output_hyper_connection_head: &'a HyperConnectionHead,
    pub(crate) layers: &'a [MlaHyperMoeLayer],
    pub(crate) kernel_plan: &'a ModelKernelPlan,
    pub(crate) mtp: Option<CudaTransformerMtpSource<'a, E>>,
}

#[cfg(feature = "cuda")]
pub(crate) type CudaTransformerRuntimeHandle = Rc<RefCell<CudaTransformerRuntime>>;

#[cfg(feature = "cuda")]
pub(crate) struct CudaTransformerRuntime {
    pub(crate) ops: Rc<cuda_linear::CudaOperators>,
    pub(crate) mla: MlaExecution,
    completion_hub: CompletionHub,
    profile: bool,
    metrics: CudaTransformerMetrics,
    managed_experts: bool,
    routed_moe: Option<RoutedMoeExecution>,
    prepared: PreparedDecoderAttachment<PreparedCudaTransformer>,
    embedding_token_ids: HashMap<usize, cuda_attention::CudaI32HostMirror>,
    compact_i32_mirrors: HashMap<usize, Vec<cuda_attention::CudaI32HostMirror>>,
    output_head_logits: HashMap<(usize, usize), cuda_linear::CudaF32Buffer>,
    output_head_linear_workspaces: HashMap<usize, cuda_linear::CudaArtifactLinearWorkspace>,
    output_head_indices: HashMap<usize, cuda_attention::CudaI32Buffer>,
    output_head_values: HashMap<usize, cuda_linear::CudaF32Buffer>,
}

/// Device-side HyperConnection weights shared by reference.
#[cfg(feature = "cuda")]
pub(crate) type CudaHyperConnection =
    Rc<PreparedHyperConnection<cuda_linear::CudaF32Buffer, cuda_linear::CudaF32Buffer>>;

/// Prepared MLA/HyperConnection/routed-MoE layer composition on device.
#[cfg(feature = "cuda")]
pub(crate) type CudaPreparedMlaHyperMoeLayer = TransformerLayer<
    Connected<CudaHyperConnection, Rc<PreparedMlaWeights>>,
    Connected<CudaHyperConnection, Rc<PreparedRoutedMoe>>,
>;

/// Output HC reduction, final norm, and LM head pipeline on device.
#[cfg(feature = "cuda")]
type CudaOutputPipeline = OutputPipeline<
    HyperReduction<PreparedHyperConnectionHead<cuda_linear::CudaF32Buffer>>,
    cuda_linear::CudaF32Buffer,
    cuda_linear::CudaArtifactLinearHandle,
>;

/// Fully prepared CUDA transformer composition with the MTP target tap.
#[cfg(feature = "cuda")]
type CudaPreparedTransformerComposition = PreparedTransformer<
    cuda_linear::CudaArtifactLinearHandle,
    CudaPreparedMlaHyperMoeLayer,
    CudaOutputPipeline,
    MtpTap,
>;

/// Prepared MTP attachment with device-side stage backbones and heads.
#[cfg(feature = "cuda")]
type CudaPreparedMtpAttachment = PreparedMtpAttachment<
    CudaPreparedMlaHyperMoeLayer,
    PreparedHyperConnectionHead<cuda_linear::CudaF32Buffer>,
    cuda_linear::CudaF32Buffer,
    PreparedCudaLinear,
    ModelKernelPlan,
>;

#[cfg(feature = "cuda")]
pub struct PreparedCudaTransformer {
    transformer: CudaPreparedTransformerComposition,
    kernel_plan: ModelKernelPlan,
    mtp: Option<CudaPreparedMtpAttachment>,
}

#[cfg(feature = "cuda")]
impl PreparedCudaTransformer {
    fn embedding(&self) -> &cuda_linear::CudaArtifactLinearHandle {
        self.transformer.embedding()
    }

    fn output_hyper_connection_head(
        &self,
    ) -> &PreparedHyperConnectionHead<cuda_linear::CudaF32Buffer> {
        self.transformer.output().reduction().head()
    }

    fn output_norm(&self) -> &cuda_linear::CudaF32Buffer {
        self.transformer.output().norm()
    }

    fn output_head(&self) -> &cuda_linear::CudaArtifactLinearHandle {
        self.transformer.output().head()
    }

    fn layers(&self) -> &[CudaPreparedMlaHyperMoeLayer] {
        self.transformer.layers()
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CudaTransformerCompileProfile {
    pub globals_us: u64,
    pub embedding_us: u64,
    pub output_head_us: u64,
    pub target_layers_us: u64,
    pub attachment_us: u64,
    pub total_us: u64,
}

#[cfg(feature = "cuda")]
impl CudaTransformerRuntime {
    pub(crate) fn mtp_input_device_into(
        &self,
        anchor_token_id: u32,
        output: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<()> {
        let image = self.prepared()?;
        let mtp = image.mtp.as_ref().ok_or_else(|| Error::Model {
            message: "CUDA transformer Proposal image is missing".into(),
        })?;
        self.ops.proposal_embedding_hc_from_resident_bf16_into(
            image.embedding(),
            anchor_token_id,
            mtp.config.noise_token_id.ok_or_else(|| Error::Model {
                message: "CUDA transformer MTP noise token is missing".into(),
            })?,
            mtp.config.block_size,
            image.output_hyper_connection_head().config().hc_mult,
            output,
        )
    }
    pub(crate) fn allocate_mtp_head_buffers(&self) -> Result<CudaMtpHeadBuffers> {
        const PARTIAL_CAPACITY: usize = 64;
        let image = self.prepared()?;
        let mtp = image.mtp.as_ref().ok_or_else(|| Error::Model {
            message: "CUDA transformer Proposal image is missing".into(),
        })?;
        let output_shape = image.output_head().shape();
        let vocab = output_shape.out_features();
        let hidden = output_shape.in_features();
        let markov_rank = mtp.heads.markov_embedding.handle.shape().in_features();
        if mtp.config.block_size != cuda_linear::PROPOSAL_ROWS
            || hidden != image.output_hyper_connection_head().config().hidden_size
            || mtp.heads.markov_embedding.handle.shape().out_features() != vocab
            || mtp.heads.markov_output.handle.shape().out_features() != vocab
            || mtp.heads.markov_output.handle.shape().in_features() != markov_rank
            || mtp.heads.confidence.handle.shape().out_features() != 1
            || mtp.heads.confidence.handle.shape().in_features()
                != hidden.saturating_add(markov_rank)
        {
            return Err(Error::Model {
                message: format!(
                    "CUDA transformer proposal-head shape mismatch: output={output_shape:?} block={} w1={:?} w2={:?} confidence={:?}",
                    mtp.config.block_size,
                    mtp.heads.markov_embedding.handle.shape(),
                    mtp.heads.markov_output.handle.shape(),
                    mtp.heads.confidence.handle.shape()
                ),
            });
        }
        Ok(CudaMtpHeadBuffers {
            workspace: self.ops.proposal_head_workspace(
                mtp.config.block_size,
                hidden,
                vocab,
                PARTIAL_CAPACITY,
            )?,
        })
    }
    pub(crate) fn capture_mtp_target_tap_from_device(
        &self,
        target_layer: usize,
        hc_state: &cuda_linear::CudaF32Buffer,
        rows: usize,
        target_taps: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<bool> {
        let image = self.prepared()?;
        let mtp = image.mtp.as_ref().ok_or_else(|| Error::Model {
            message: "CUDA transformer Proposal image is missing".into(),
        })?;
        let Some(tap_slot) = mtp
            .config
            .target_layer_ids
            .iter()
            .position(|layer| *layer == target_layer)
        else {
            return Ok(false);
        };
        self.ops.hc_mean_scatter_from_device_into(
            hc_state,
            rows,
            image.output_hyper_connection_head().config().hc_mult,
            image.output_hyper_connection_head().config().hidden_size,
            tap_slot,
            mtp.config.target_layer_ids.len(),
            target_taps,
        )?;
        Ok(true)
    }
    pub(crate) fn mtp_main_project_norm_device_into(
        &self,
        rows: usize,
        buffers: &mut MlaProposalMainBuffers,
    ) -> Result<()> {
        let descriptor =
            self.require_mtp_inference_descriptor(0, KernelOperation::MainProjectNorm)?;
        if !descriptor.is_provider_managed() {
            return Err(Error::Model {
                message: format!(
                    "invalid Proposal main-project/norm provider binding: {:?}",
                    descriptor.kernel
                ),
            });
        }
        let image = self.prepared()?;
        let stage_zero = image
            .mtp
            .as_ref()
            .and_then(|mtp| mtp.stages.first())
            .ok_or_else(|| Error::Model {
                message: "CUDA transformer Proposal stage zero is missing".into(),
            })?;
        let projection = stage_zero
            .main_projection
            .as_ref()
            .ok_or_else(|| Error::Model {
                message: "CUDA transformer Proposal stage-zero projection is missing".into(),
            })?;
        let norm = stage_zero.main_norm.as_ref().ok_or_else(|| Error::Model {
            message: "CUDA transformer Proposal stage-zero norm is missing".into(),
        })?;
        self.ops.artifact_main_project_norm_into(
            &projection.handle,
            norm,
            &buffers.target_taps,
            rows,
            image.output_hyper_connection_head().config().norm_eps,
            &mut buffers.activation,
            &mut buffers.inv_rms,
            &mut buffers.main_x,
        )
    }
    pub(crate) fn mtp_head_device_into(
        &self,
        anchor_token_id: u32,
        hc_state: &cuda_linear::CudaF32Buffer,
        buffers: &mut CudaMtpHeadBuffers,
    ) -> Result<()> {
        let descriptor = self.require_mtp_inference_descriptor(0, KernelOperation::ProposalHead)?;
        if !descriptor.is_provider_managed() {
            return Err(Error::Model {
                message: format!(
                    "invalid Proposal-head provider binding: {:?}",
                    descriptor.kernel
                ),
            });
        }
        let image = self.prepared()?;
        let mtp = image.mtp.as_ref().ok_or_else(|| Error::Model {
            message: "CUDA transformer Proposal image is missing".into(),
        })?;
        let output_shape = image.output_head().shape();
        let markov_rank = mtp.heads.markov_embedding.handle.shape().in_features();
        let hyper_connection = &mtp.heads.hyper_connection_head;
        self.ops.artifact_proposal_head_into(
            hc_state,
            hyper_connection.weights().function_row_major(),
            hyper_connection.weights().scale(),
            hyper_connection.weights().base(),
            &mtp.heads.norm,
            image.output_head(),
            &mtp.heads.markov_embedding.handle,
            &mtp.heads.markov_output.handle,
            &mtp.heads.confidence.handle,
            anchor_token_id,
            cuda_linear::ProposalHeadLayout {
                rows: mtp.config.block_size,
                hc: hyper_connection.config().hc_mult,
                hidden: output_shape.in_features(),
                vocab: output_shape.out_features(),
                markov_rank,
                partial_capacity: 64,
                hc_eps: hyper_connection.config().eps,
                norm_eps: hyper_connection.config().norm_eps,
            },
            &mut buffers.workspace,
        )
    }
    pub(crate) fn begin_mtp_head_result_download(
        &self,
        buffers: &mut CudaMtpHeadBuffers,
    ) -> Result<cuda_attention::CudaI32HostDownload> {
        self.ops
            .begin_proposal_head_result_download(&mut buffers.workspace)
    }
    pub(crate) fn poll_mtp_head_result(
        &self,
        buffers: &mut CudaMtpHeadBuffers,
        download: &cuda_attention::CudaI32HostDownload,
    ) -> Result<Option<Vec<i32>>> {
        self.ops
            .poll_proposal_head_result(&mut buffers.workspace, download)
    }
    pub(crate) fn decode_mtp_head_result(&self, compact: Vec<i32>) -> Result<(Vec<u32>, Vec<f32>)> {
        let rows = cuda_linear::PROPOSAL_ROWS;
        if compact.len() != 1 + 2 * rows || compact[0] != 0 {
            return Err(Error::Execution {
                message: format!(
                    "CUDA transformer proposal compact proposal-head result is invalid: {compact:?}"
                ),
            });
        }
        let token_ids = compact[1..1 + rows]
            .iter()
            .copied()
            .map(|token| {
                u32::try_from(token).map_err(|_| Error::Execution {
                    message: format!("CUDA transformer proposal emitted invalid token {token}"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let confidence = compact[1 + rows..]
            .iter()
            .map(|bits| f32::from_bits(*bits as u32))
            .collect();
        Ok((token_ids, confidence))
    }
}

#[cfg(feature = "cuda")]
impl CudaTransformerRuntime {
    pub(crate) fn pinned_host_allocator(&self) -> cuda_moe::CudaPinnedHostAllocator {
        self.ops.pinned_host_allocator()
    }
    pub(crate) fn memory_info(&self) -> Result<(usize, usize)> {
        self.ops.memory_info()
    }
    pub(crate) fn compute_stream_authority(&self) -> cuda_moe::CudaComputeStreamAuthority {
        self.ops.compute_stream_authority()
    }

    pub(crate) fn new_with_completion_hub(
        config: CudaTransformerRuntimeConfig,
        completion_hub: CompletionHub,
        prepared: PreparedDecoderAttachment<PreparedCudaTransformer>,
    ) -> Result<Self> {
        let ops = Rc::new(cuda_linear::CudaOperators::new()?);
        let mla = MlaExecution::new(Rc::clone(&ops), config.profile, config.profile_sync);
        Ok(Self {
            ops,
            mla,
            completion_hub,
            profile: config.profile,
            metrics: CudaTransformerMetrics::default(),
            managed_experts: config.managed_experts,
            routed_moe: None,
            prepared,
            embedding_token_ids: HashMap::new(),
            compact_i32_mirrors: HashMap::new(),
            output_head_logits: HashMap::new(),
            output_head_linear_workspaces: HashMap::new(),
            output_head_indices: HashMap::new(),
            output_head_values: HashMap::new(),
        })
    }

    pub(crate) fn configure_expert_subsystem(
        &mut self,
        subsystem: CudaSharedExpertSubsystem,
    ) -> Result<()> {
        if self.routed_moe.is_some() {
            return Err(Error::Execution {
                message: "CUDA routed-MoE execution is already attached".into(),
            });
        }
        self.routed_moe = Some(RoutedMoeExecution::new(
            Rc::clone(&self.ops),
            subsystem,
            self.completion_hub.clone(),
        ));
        Ok(())
    }

    /// Reduces packed HyperConnection rows to final hidden rows through the
    /// prepared output HC head retained by the compiled execution image.
    pub(crate) fn hc_head_output_rows_device_into(
        &self,
        hidden: &cuda_linear::CudaF32Buffer,
        rows: usize,
        output: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<()> {
        self.prepared()?
            .output_hyper_connection_head()
            .reduce(&self.ops, hidden, rows, output)
    }

    pub(crate) fn shutdown(&mut self) -> Result<()> {
        self.ops.sync_stream()?;
        {
            self.output_head_logits.clear();
            self.output_head_linear_workspaces.clear();
            self.output_head_indices.clear();
            self.output_head_values.clear();
            self.embedding_token_ids.clear();
        }
        self.prepared.shutdown();
        self.compact_i32_mirrors.clear();
        self.mla.clear();
        self.routed_moe = None;
        Ok(())
    }

    fn upload_resident_bf16_matrix(
        &self,
        matrix: &CheckpointMatrixSlice,
    ) -> Result<cuda_linear::CudaArtifactLinearHandle> {
        const CHUNK_BYTES: usize = 16 * 1024 * 1024;
        if matrix.slice.dtype != CheckpointDType::Bf16 {
            return Err(Error::Model {
                message: format!(
                    "resident BF16 matrix '{}' has dtype {:?}",
                    matrix.slice.name, matrix.slice.dtype
                ),
            });
        }
        let expected_bytes = matrix
            .rows
            .checked_mul(matrix.cols)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::Model {
                message: "resident BF16 matrix size overflow".into(),
            })?;
        if matrix.slice.bytes != expected_bytes as u64 {
            return Err(Error::Model {
                message: format!(
                    "resident BF16 matrix '{}' byte mismatch: slice={} expected={expected_bytes}",
                    matrix.slice.name, matrix.slice.bytes
                ),
            });
        }
        let shape = cuda_linear::CudaArtifactLinearShape::Bf16Bytes {
            out_features: matrix.rows,
            in_features: matrix.cols,
        };
        let mut handle = self.ops.allocate_artifact_linear_device(shape)?;
        let mut file = File::open(&matrix.slice.path).map_err(|error| Error::Model {
            message: format!(
                "open resident BF16 matrix '{}': {error}",
                matrix.slice.path.display()
            ),
        })?;
        file.seek(SeekFrom::Start(matrix.slice.offset))
            .map_err(|error| Error::Model {
                message: format!(
                    "seek resident BF16 matrix '{}': {error}",
                    matrix.slice.path.display()
                ),
            })?;
        let mut chunk = vec![0u8; CHUNK_BYTES.min(expected_bytes)];
        let mut offset = 0usize;
        while offset < expected_bytes {
            let bytes = chunk.len().min(expected_bytes - offset);
            file.read_exact(&mut chunk[..bytes])
                .map_err(|error| Error::Model {
                    message: format!(
                        "read resident BF16 matrix '{}' at {offset}: {error}",
                        matrix.slice.name
                    ),
                })?;
            self.ops.overwrite_artifact_linear_weight_range(
                &mut handle,
                offset,
                &chunk[..bytes],
            )?;
            offset += bytes;
        }
        Ok(handle)
    }
    fn upload_prepared_linear(&self, linear: &LinearWeight) -> Result<PreparedCudaLinear> {
        Ok(PreparedCudaLinear::new(
            self.upload_linear(linear)?,
            linear.execution.activation_quantization,
        ))
    }
    fn upload_compressor_resources(
        &self,
        payload: &MlaCompressor,
    ) -> Result<PreparedMlaCompressor> {
        Ok(PreparedMlaCompressor {
            ape: self.ops.upload_f32_buffer(&payload.ape)?,
            norm: self.upload_norm_weight(&payload.norm)?,
            kv: self.upload_prepared_linear(&payload.wkv)?,
            gate: self.upload_prepared_linear(&payload.wgate)?,
        })
    }
    fn upload_prepared_layer(
        &mut self,
        layer: &MlaHyperMoeLayer,
    ) -> Result<CudaPreparedMlaHyperMoeLayer> {
        let layer_id = layer.layer;
        let main_compressor = layer
            .attention
            .compressed
            .as_ref()
            .map(|compressed| self.upload_compressor_resources(&compressed.compressor))
            .transpose()?;
        let indexer = layer
            .attention
            .compressed
            .as_ref()
            .and_then(|compressed| compressed.indexer.as_ref())
            .map(|indexer| {
                Ok::<_, Error>(PreparedMlaIndexer {
                    compressor: self.upload_compressor_resources(&indexer.compressor)?,
                    query: self.upload_prepared_linear(&indexer.wq_b)?,
                    weights: self.upload_prepared_linear(&indexer.weights_proj)?,
                })
            })
            .transpose()?;
        let mla = Rc::new(PreparedMlaWeights {
            query_a: self.upload_prepared_linear(&layer.attention.payload.query_a)?,
            query_b: self.upload_prepared_linear(&layer.attention.payload.query_b)?,
            key_value: self.upload_prepared_linear(&layer.attention.payload.key_value)?,
            query_norm: self.upload_norm_weight(&layer.attention.payload.query_norm)?,
            key_value_norm: self.upload_norm_weight(&layer.attention.payload.key_value_norm)?,
            attention_sink: self
                .ops
                .upload_f32_buffer(&layer.attention.payload.attention_sink)?,
            main_compressor,
            indexer,
            output: PreparedMlaOutput {
                a: self.upload_prepared_linear(&layer.attention.payload.output_a)?,
                b: self.upload_prepared_linear(&layer.attention.payload.output_b)?,
            },
        });
        self.mla.install(layer_id, Rc::clone(&mla))?;
        let feed_forward = Rc::new(PreparedRoutedMoe::prepare(
            &layer.feed_forward,
            &self.ops,
            |linear| self.upload_linear(linear),
        )?);
        let hyper_connection = Rc::new(PreparedHyperConnection::prepare(
            &self.ops,
            &layer.hyper_connection,
        )?);
        Ok(TransformerLayer::new(
            layer_id,
            Connected::new(Rc::clone(&hyper_connection), mla),
            Connected::new(hyper_connection, feed_forward),
        ))
    }
    fn upload_prepared_mtp_attachment<E>(
        &mut self,
        source: Option<CudaTransformerMtpSource<'_, E>>,
    ) -> Result<Option<CudaPreparedMtpAttachment>> {
        let Some(source) = source else {
            return Ok(None);
        };
        let mtp = source.attachment;
        let transformer_kernel_plan = source.kernel_plan;
        if transformer_kernel_plan.layers.len() != mtp.stages.len() {
            return Err(Error::Model {
                message: format!(
                    "CUDA transformer MTP kernel plan has {} layers for {} prepared stages",
                    transformer_kernel_plan.layers.len(),
                    mtp.stages.len()
                ),
            });
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(mtp.stages.len())
            .map_err(|error| Error::Model {
                message: format!(
                    "CUDA transformer MTP image allocation failed for {} stages: {error}",
                    mtp.stages.len()
                ),
            })?;
        for (stage, layer) in mtp.stages.iter().enumerate() {
            if layer.index != stage {
                return Err(Error::Model {
                    message: format!(
                        "CUDA transformer MTP stage identity mismatch: slot={stage} checkpoint_stage={}",
                        layer.index
                    ),
                });
            }
            if layer.backbone.layer != layer.execution_layer {
                return Err(Error::Model {
                    message: format!(
                        "CUDA transformer MTP execution identity mismatch: stage={stage} transformer_layer={} execution_layer={}",
                        layer.backbone.layer, layer.execution_layer
                    ),
                });
            }
            layers.push(PreparedMtpStage {
                execution_layer: layer.execution_layer,
                backbone: self.upload_prepared_layer(&layer.backbone)?,
                main_projection: layer
                    .main_projection
                    .as_ref()
                    .map(|linear| self.upload_prepared_linear(linear))
                    .transpose()?,
                main_norm: layer
                    .main_norm
                    .as_deref()
                    .map(|norm| self.upload_norm_weight(norm))
                    .transpose()?,
            });
        }
        let prediction_heads = mtp.heads.as_ref().ok_or_else(|| Error::Model {
            message: "CUDA transformer MTP image requires prediction heads".into(),
        })?;
        let heads = PreparedMtpHeads {
            hyper_connection_head: PreparedHyperConnectionHead::prepare(
                &self.ops,
                &prediction_heads.hyper_connection_head,
            )?,
            norm: self.upload_norm_weight(&prediction_heads.norm)?,
            markov_embedding: self.upload_prepared_linear(&prediction_heads.markov_embedding)?,
            markov_output: self.upload_prepared_linear(&prediction_heads.markov_output)?,
            confidence: self.upload_prepared_linear(&prediction_heads.confidence)?,
        };
        mtp.config.protocol()?;
        Ok(Some(PreparedMtpAttachment {
            config: mtp.config.clone(),
            stages: layers.into_boxed_slice(),
            heads,
            compiled_plan: transformer_kernel_plan.clone(),
        }))
    }
    pub(crate) fn compile<E>(
        &mut self,
        generation: u64,
        source: CudaTransformerSource<'_, E>,
    ) -> Result<CudaTransformerCompileProfile> {
        let layers = source.layers;
        let kernel_plan = source.kernel_plan;
        if kernel_plan.layers.len() != layers.len() {
            return Err(Error::Model {
                message: format!(
                    "CUDA transformer kernel plan has {} layers for {} prepared layers",
                    kernel_plan.layers.len(),
                    layers.len()
                ),
            });
        }
        let attachment = self.prepared.clone();
        if attachment.generation() != generation {
            return Err(Error::Internal {
                message: format!(
                    "CUDA transformer prepared generation mismatch: attachment={} requested={generation}",
                    attachment.generation()
                ),
            });
        }
        if attachment.is_installed() {
            return Err(Error::Internal {
                message: "CUDA transformer prepared state is already installed".into(),
            });
        }
        let total_start = Instant::now();
        let phase_start = Instant::now();
        let hyper_connection_head =
            PreparedHyperConnectionHead::prepare(&self.ops, source.output_hyper_connection_head)?;
        let output_norm = self.upload_norm_weight(source.output_norm)?;
        let globals_us = duration_us(phase_start.elapsed());
        let phase_start = Instant::now();
        let embedding = self.upload_resident_bf16_matrix(source.embedding)?;
        let embedding_us = duration_us(phase_start.elapsed());
        let phase_start = Instant::now();
        let output_head = self.upload_resident_bf16_matrix(source.output_head)?;
        let output_head_us = duration_us(phase_start.elapsed());
        let phase_start = Instant::now();
        let mut prepared_layers = Vec::new();
        prepared_layers
            .try_reserve_exact(layers.len())
            .map_err(|error| Error::Model {
                message: format!(
                    "CUDA transformer prepared layer allocation failed for {} layers: {error}",
                    layers.len()
                ),
            })?;
        for (layer_index, layer) in layers.iter().enumerate() {
            if layer.layer != layer_index {
                return Err(Error::Model {
                    message: format!(
                        "CUDA transformer prepared layer identity mismatch: slot={layer_index} layer={}",
                        layer.layer
                    ),
                });
            }
            prepared_layers.push(self.upload_prepared_layer(layer)?);
        }
        let target_layers_us = duration_us(phase_start.elapsed());
        let phase_start = Instant::now();
        let prepared_mtp = self.upload_prepared_mtp_attachment(source.mtp)?;
        let attachment_us = duration_us(phase_start.elapsed());
        let target_taps = prepared_mtp
            .as_ref()
            .map(|mtp| mtp.config.target_layer_ids.clone())
            .unwrap_or_default();
        attachment.install(PreparedCudaTransformer {
            transformer: PreparedTransformer::new(
                embedding,
                prepared_layers,
                OutputPipeline::new(
                    HyperReduction::new(hyper_connection_head),
                    output_norm,
                    output_head,
                ),
                MtpTap::new(target_taps)?,
            ),
            kernel_plan: kernel_plan.clone(),
            mtp: prepared_mtp,
        })?;

        Ok(CudaTransformerCompileProfile {
            globals_us,
            embedding_us,
            output_head_us,
            target_layers_us,
            attachment_us,
            total_us: duration_us(total_start.elapsed()),
        })
    }
    fn prepared(&self) -> Result<&PreparedCudaTransformer> {
        self.prepared.get()
    }
    fn require_inference_descriptor(
        &self,
        execution_layer: usize,
        operation: KernelOperation,
    ) -> Result<LaunchDescriptor> {
        let image = self.prepared()?;
        let descriptor = image
            .kernel_plan
            .layer(execution_layer)
            .and_then(|plan| plan.operation(operation, ExecutionMode::Inference))
            .copied()
            .or_else(|| {
                let mtp = image.mtp.as_ref()?;
                let stage = mtp
                    .stages
                    .iter()
                    .position(|layer| layer.execution_layer == execution_layer)?;
                mtp.compiled_plan
                    .layer(stage)
                    .and_then(|plan| plan.operation(operation, ExecutionMode::Inference))
                    .copied()
            });
        descriptor.ok_or_else(|| Error::Model {
            message: format!(
                "inference operation {operation:?} is missing for execution layer={execution_layer}"
            ),
        })
    }
    fn require_mtp_inference_descriptor(
        &self,
        stage: usize,
        operation: KernelOperation,
    ) -> Result<LaunchDescriptor> {
        self.prepared()?
        .mtp
        .as_ref()
        .and_then(|mtp| mtp.compiled_plan.layer(stage))
        .and_then(|plan| plan.operation(operation, ExecutionMode::Inference))
        .copied()
        .ok_or_else(|| Error::Model {
            message: format!(
                "CUDA transformer MTP stage {stage} is missing inference operation {operation:?}"
            ),
        })
    }
    fn require_inference_operation(&self, layer: usize, operation: KernelOperation) -> Result<()> {
        let descriptor = self.require_inference_descriptor(layer, operation)?;
        if descriptor.kernel.operation != operation
            || descriptor.kernel.mode != ExecutionMode::Inference
            || !descriptor.is_provider_managed()
        {
            return Err(Error::Model {
                message: format!(
                    "invalid CUDA inference binding for layer={layer} operation={operation:?}: {:?}",
                    descriptor.kernel
                ),
            });
        }
        Ok(())
    }
    fn prepared_layer(&self, execution_layer: usize) -> Result<&CudaPreparedMlaHyperMoeLayer> {
        let image = self.prepared()?;
        if let Some(layer) = image.layers().get(execution_layer) {
            return Ok(layer);
        }
        image
            .mtp
            .as_ref()
            .and_then(|mtp| {
                mtp.stages
                    .iter()
                    .find(|layer| layer.execution_layer == execution_layer)
            })
            .map(|layer| &layer.backbone)
            .ok_or_else(|| Error::Model {
                message: format!(
                    "CUDA transformer execution layer {execution_layer} is not prepared"
                ),
            })
    }
    pub(crate) fn routed_moe_execution(&mut self) -> Result<&mut RoutedMoeExecution> {
        self.routed_moe.as_mut().ok_or_else(|| Error::Execution {
            message: "CUDA routed-MoE execution is not attached".into(),
        })
    }

    pub(crate) fn routed_moe_parts(
        &mut self,
        execution_layer: usize,
    ) -> Result<(Rc<PreparedRoutedMoe>, &mut RoutedMoeExecution)> {
        let prepared = Rc::clone(self.prepared_layer(execution_layer)?.feed_forward().block());
        let execution = self.routed_moe.as_mut().ok_or_else(|| Error::Execution {
            message: "CUDA routed-MoE execution is not attached".into(),
        })?;
        Ok((prepared, execution))
    }

    pub(crate) fn upload_norm_weight(&self, weight: &[f32]) -> Result<cuda_linear::CudaF32Buffer> {
        self.ops.upload_norm_weight(weight)
    }

    pub(crate) fn upload_linear(
        &self,
        linear: &LinearWeight,
    ) -> Result<cuda_linear::CudaArtifactLinearHandle> {
        match linear.format {
            LinearWeightFormat::F32 {
                out_features,
                in_features,
            } => self
                .ops
                .upload_f32_linear(&linear.weight.bytes, out_features, in_features),
            LinearWeightFormat::Bf16 {
                out_features,
                in_features,
            } => self
                .ops
                .upload_bf16_linear(&linear.weight.bytes, out_features, in_features),
            LinearWeightFormat::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m,
                block_k,
            } => {
                let scale = linear.scale.as_ref().ok_or_else(|| Error::Model {
                    message: format!(
                        "artifact linear {:?} CUDA FP8 weight is missing E8M0 scale tensor",
                        linear.role
                    ),
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
            LinearWeightFormat::Fp4E2M1PackedWithE8M0Scale {
                out_features,
                in_features,
                block_size: 32,
            } => {
                let scale = linear.scale.as_ref().ok_or_else(|| Error::Model {
                    message: format!(
                        "artifact linear {:?} CUDA FP4 weight is missing E8M0 scale tensor",
                        linear.role
                    ),
                })?;
                self.ops.upload_fp4_e2m1_e8m0_linear(
                    &linear.weight.bytes,
                    &scale.bytes,
                    out_features,
                    in_features,
                    self.managed_experts,
                )
            }
            LinearWeightFormat::Fp4E2M1PackedWithE8M0Scale { block_size, .. } => {
                Err(Error::Model {
                    message: format!(
                        "artifact linear {:?} CUDA FP4 block_size {block_size} is unsupported (expected 32)",
                        linear.role
                    ),
                })
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}
/// Kernel requirements for one MLA/HyperConnection/routed-MoE layer.
#[cfg(feature = "cuda")]
pub(crate) fn mla_hyper_moe_kernel_requirements(
    layer: &MlaHyperMoeLayer,
) -> Result<LayerKernelRequirements> {
    let mut value = LayerKernelRequirements::default();
    for operation in [
        KernelOperation::AttentionHcPre,
        KernelOperation::FeedForwardHcPre,
        KernelOperation::SharedFfn,
        KernelOperation::GroupedFp4Moe,
        KernelOperation::MlaOutput,
    ] {
        value.require(OperationRequirement::new(
            operation,
            ExecutionMode::Inference,
        ));
    }
    value.add_linear_bundle(fp8_bundle(
        KernelOperation::MlaQueryAKv,
        [
            &layer.attention.payload.query_a,
            &layer.attention.payload.key_value,
        ],
    )?);
    value.add_linear_bundle(fp8_single(
        KernelOperation::MlaQueryB,
        &layer.attention.payload.query_b,
    )?);
    if let Some(compressed) = layer.attention.compressed.as_ref() {
        value.add_linear_bundle(bf16_bundle(
            KernelOperation::MainCompressorProjection,
            [&compressed.compressor.wkv, &compressed.compressor.wgate],
        )?);
        if let Some(indexer) = compressed.indexer.as_ref() {
            value.add_linear_bundle(bf16_bundle(
                KernelOperation::IndexerCompressorProjection,
                [&indexer.compressor.wkv, &indexer.compressor.wgate],
            )?);
        }
    }
    Ok(value)
}

/// Compiles the target and optional MTP kernel plans for an MLA/HC/MoE model.
#[cfg(feature = "cuda")]
pub(crate) fn compile_mla_hyper_moe_plans(
    layers: &[MlaHyperMoeLayer],
    proposal: Option<&MtpAttachment<MlaHyperMoeLayer, Arc<ExpertSourceCatalog>>>,
) -> Result<(ModelKernelPlan, Option<ModelKernelPlan>)> {
    let base = layers
        .iter()
        .map(mla_hyper_moe_kernel_requirements)
        .collect::<Result<Vec<_>>>()?;
    let base = ferrule_backend::cuda::compile_model_plan(&base)?;
    let proposal = proposal
        .map(|attachment| {
            let requirements = attachment
                .stages
                .iter()
                .enumerate()
                .map(|(stage, layer)| {
                    let mut value = mla_hyper_moe_kernel_requirements(&layer.backbone)?;
                    value.require(OperationRequirement::new(
                        KernelOperation::HybridMlaAttention,
                        ExecutionMode::Inference,
                    ));
                    if stage == 0 {
                        for operation in [
                            KernelOperation::ProposalHead,
                            KernelOperation::MainProjectNorm,
                        ] {
                            value.require(OperationRequirement::new(
                                operation,
                                ExecutionMode::Inference,
                            ));
                        }
                    }
                    Ok(value)
                })
                .collect::<Result<Vec<_>>>()?;
            ferrule_backend::cuda::compile_model_plan(&requirements)
        })
        .transpose()?;
    Ok((base, proposal))
}

#[cfg(feature = "cuda")]
fn fp8_bundle(
    operation: KernelOperation,
    [first, second]: [&LinearWeight; 2],
) -> Result<LinearBundleRequirement> {
    let (
        LinearWeightFormat::Fp8E4M3WithE8M0Scale {
            out_features: first_out,
            in_features: first_in,
            ..
        },
        LinearWeightFormat::Fp8E4M3WithE8M0Scale {
            out_features: second_out,
            in_features: second_in,
            ..
        },
    ) = (&first.format, &second.format)
    else {
        return Err(Error::Model {
            message: format!("{operation:?} requires FP8"),
        });
    };
    if first_in != second_in {
        return Err(Error::Model {
            message: format!("{operation:?} input mismatch"),
        });
    }
    Ok(LinearBundleRequirement::new(
        operation,
        ExecutionMode::Inference,
        *first_in,
        [*first_out, *second_out],
        WeightLayout::Fp8E4m3BlockScaled,
    ))
}

#[cfg(feature = "cuda")]
fn fp8_single(operation: KernelOperation, value: &LinearWeight) -> Result<LinearBundleRequirement> {
    let LinearWeightFormat::Fp8E4M3WithE8M0Scale {
        out_features,
        in_features,
        ..
    } = &value.format
    else {
        return Err(Error::Model {
            message: format!("{operation:?} requires FP8"),
        });
    };
    Ok(LinearBundleRequirement::new(
        operation,
        ExecutionMode::Inference,
        *in_features,
        [*out_features],
        WeightLayout::Fp8E4m3BlockScaled,
    ))
}

#[cfg(feature = "cuda")]
fn bf16_bundle(
    operation: KernelOperation,
    [first, second]: [&LinearWeight; 2],
) -> Result<LinearBundleRequirement> {
    let (
        LinearWeightFormat::Bf16 {
            out_features: first_out,
            in_features: first_in,
        },
        LinearWeightFormat::Bf16 {
            out_features: second_out,
            in_features: second_in,
        },
    ) = (&first.format, &second.format)
    else {
        return Err(Error::Model {
            message: format!("{operation:?} requires BF16"),
        });
    };
    if first_in != second_in {
        return Err(Error::Model {
            message: format!("{operation:?} input mismatch"),
        });
    }
    Ok(LinearBundleRequirement::new(
        operation,
        ExecutionMode::Inference,
        *first_in,
        [*first_out, *second_out],
        WeightLayout::Bf16RowMajor,
    ))
}

#[cfg(feature = "cuda")]
pub(crate) use cuda_runtime::CudaOutputHeadTopKDownload;

#[cfg(feature = "cuda")]
pub(crate) mod cuda_runtime {

    use super::{CudaTransformerBuffers, CudaTransformerRuntime};
    use crate::transformer::attention::mla::MlaProposalMainBuffers;

    use crate::runner::{TokenLogit, completion_notify_callback};
    use ferrule_backend::cuda::operators::{attention as cuda_attention, linear as cuda_linear};
    use ferrule_common::{Error, Result};

    use std::time::{Duration, Instant};

    struct CudaCompactI32Download {
        compact_len: usize,
        mirror: Option<cuda_attention::CudaI32HostMirror>,
        download: cuda_attention::CudaI32HostDownload,
        callback_armed: bool,
    }
    pub(crate) struct CudaOutputHeadTopKDownload {
        rows: usize,
        top_k: usize,
        vocab: usize,
        compact: CudaCompactI32Download,
        topk_start: Option<Instant>,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CudaCompactDownloadPollAction {
        Wait,
        ArmAndWait,
        Consume,
    }
    const fn compact_download_poll_action(
        ready: bool,
        callback_armed: bool,
    ) -> CudaCompactDownloadPollAction {
        if ready {
            CudaCompactDownloadPollAction::Consume
        } else if callback_armed {
            CudaCompactDownloadPollAction::Wait
        } else {
            CudaCompactDownloadPollAction::ArmAndWait
        }
    }
    fn duration_us(d: Duration) -> u64 {
        d.as_micros().min(u128::from(u64::MAX)) as u64
    }
    #[inline]
    fn profile_start(enabled: bool) -> Option<Instant> {
        enabled.then(Instant::now)
    }
    #[inline]
    fn record_profile_duration(stat: &mut u64, start: Option<Instant>) {
        if let Some(start) = start {
            *stat = stat.saturating_add(duration_us(start.elapsed()));
        }
    }
    fn decode_output_head_topk_rows(
        compact: &[i32],
        batch_rows: usize,
        top_k: usize,
        vocab: usize,
    ) -> Result<Vec<Vec<TokenLogit>>> {
        if top_k == 0 {
            if compact.is_empty() {
                return Ok((0..batch_rows).map(|_| Vec::new()).collect());
            }
            return Err(Error::Model {
                message: format!(
                    "CUDA transformer output-head compact result must be empty for k=0, got {} values",
                    compact.len()
                ),
            });
        }
        let expected_pairs = batch_rows.checked_mul(top_k).ok_or_else(|| Error::Model {
            message: "CUDA transformer output-head top-k size overflow".into(),
        })?;
        let expected_len = expected_pairs.checked_mul(2).ok_or_else(|| Error::Model {
            message: "CUDA transformer compact output-head size overflow".into(),
        })?;
        if compact.len() != expected_len {
            return Err(Error::Model {
                message: format!(
                    "CUDA transformer output-head compact result mismatch: rows={batch_rows} k={top_k} expected={expected_len} got={}",
                    compact.len()
                ),
            });
        }
        compact
            .chunks_exact(top_k * 2)
            .map(|row| {
                row.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|[token_bits, logit_bits]| {
                        let token = usize::try_from(*token_bits).map_err(|_| Error::Model {
                            message: format!(
                                "CUDA transformer output-head returned negative token index {}",
                                token_bits
                            ),
                        })?;
                        if token >= vocab {
                            return Err(Error::Model {
                                message: format!(
                                    "CUDA transformer output-head token index {token} exceeds vocab {vocab}"
                                ),
                            });
                        }
                        Ok(TokenLogit {
                            token_id: u32::try_from(token).map_err(|_| Error::Model {
                                message: format!(
                                    "CUDA transformer output-head token index {token} exceeds u32"
                                ),
                            })?,
                            logit: f32::from_bits(*logit_bits as u32),
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect()
    }
    impl CudaTransformerRuntime {
        pub(crate) fn check_arena_acquire(&self) -> Result<()> {
            if self.ops.failpoints().check_arena_acquire() {
                return Err(Error::Internal {
                    message: "deterministic failpoint: CUDA transformer arena acquire".into(),
                });
            }
            Ok(())
        }
        pub(crate) fn allocate_decode_buffers(
            &self,
            hc_len: usize,
            hidden_len: usize,
        ) -> Result<CudaTransformerBuffers> {
            Ok(CudaTransformerBuffers {
                hc_input: self.ops.zero_f32_buffer(hc_len)?,
                final_hidden: self.ops.zero_f32_buffer(hidden_len)?,
                topk_row: self.ops.zero_f32_buffer(hidden_len)?,
            })
        }
        pub(crate) fn allocate_proposal_main_buffers(
            &self,
            rows: usize,
        ) -> Result<MlaProposalMainBuffers> {
            if rows == 0 {
                return Err(Error::Model {
                    message: "CUDA transformer proposal main buffers require at least one row"
                        .into(),
                });
            }
            let (tap_count, input_size, output_size, context_kv_size) = {
                let mtp = self.prepared()?.mtp.as_ref().ok_or_else(|| Error::Model {
                    message: "CUDA transformer Proposal image is missing".into(),
                })?;
                let stage_zero = mtp.stages.first().ok_or_else(|| Error::Model {
                    message: "CUDA transformer Proposal image has no stages".into(),
                })?;
                let projection =
                    stage_zero
                        .main_projection
                        .as_ref()
                        .ok_or_else(|| Error::Model {
                            message:
                                "CUDA transformer Proposal stage zero main projection is missing"
                                    .into(),
                        })?;
                let output_size = projection.handle.shape().out_features();
                let context_kv_size = stage_zero
                    .backbone
                    .attention()
                    .block()
                    .key_value
                    .handle
                    .shape()
                    .out_features();
                for (stage, layer) in mtp.stages.iter().enumerate() {
                    let key_value = &layer.backbone.attention().block().key_value.handle;
                    if key_value.shape().in_features() != output_size
                        || key_value.shape().out_features() != context_kv_size
                        || layer.backbone.attention().block().key_value_norm.len()
                            != context_kv_size
                    {
                        return Err(Error::Model {
                            message: format!(
                                "CUDA transformer Proposal stage {stage} context-KV shape mismatch: wkv={:?} norm={} main_x={output_size} expected_kv={context_kv_size}",
                                key_value.shape(),
                                layer.backbone.attention().block().key_value_norm.len()
                            ),
                        });
                    }
                }
                (
                    mtp.config.target_layer_ids.len(),
                    projection.handle.shape().in_features(),
                    output_size,
                    context_kv_size,
                )
            };
            if tap_count == 0 || input_size % tap_count != 0 {
                return Err(Error::Model {
                    message: format!(
                        "CUDA transformer Proposal tap layout is invalid: taps={tap_count} input={input_size}"
                    ),
                });
            }
            let target_taps_len = rows.checked_mul(input_size).ok_or_else(|| Error::Model {
                message: "CUDA transformer Proposal target-tap size overflow".into(),
            })?;
            let main_x_len = rows.checked_mul(output_size).ok_or_else(|| Error::Model {
                message: "CUDA transformer Proposal main-x size overflow".into(),
            })?;
            let context_kv_len = rows
                .checked_mul(context_kv_size)
                .ok_or_else(|| Error::Model {
                    message: "CUDA transformer Proposal context-KV size overflow".into(),
                })?;
            Ok(MlaProposalMainBuffers {
                target_taps: self.ops.zero_f32_buffer(target_taps_len)?,
                positions: self.ops.zero_i32_buffer(rows)?,
                activation: self.ops.fp8_activation_pack(rows, input_size)?,
                inv_rms: self.ops.zero_f32_buffer(rows)?,
                main_x: self.ops.zero_f32_buffer(main_x_len)?,
                context_kv_raw: self.ops.zero_f32_buffer(context_kv_len)?,
                context_kv: self.ops.zero_f32_buffer(context_kv_len)?,
                context_linear_workspace: self.ops.artifact_linear_workspace(rows, output_size)?,
            })
        }
        pub(crate) fn resident_embedding_hc_rows_into(
            &mut self,
            token_ids: &[u32],
            output: &mut cuda_linear::CudaF32Buffer,
        ) -> Result<()> {
            {
                let (vocab, hc_mult) = {
                    let image = self.prepared()?;
                    (
                        image.embedding().shape().out_features(),
                        image.output_hyper_connection_head().config().hc_mult,
                    )
                };
                if token_ids.is_empty() {
                    return Err(Error::Model {
                        message:
                            "CUDA transformer packed CUDA embedding gather requires at least one token"
                                .into(),
                    });
                }
                let device_values = token_ids
                    .iter()
                    .copied()
                    .map(|token_id| {
                        if token_id as usize >= vocab {
                            return Err(Error::Model {
                                message: format!("CUDA transformer token id {token_id} exceeds resident embedding vocab {vocab}"),
                            });
                        }
                        i32::try_from(token_id).map_err(|_| Error::Model {
                            message: format!("CUDA transformer token id {token_id} exceeds the CUDA i32 token ABI"),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let rows = token_ids.len();
                if let Some(cached) = self.embedding_token_ids.get_mut(&rows) {
                    self.ops.update_i32_host_mirror(&device_values, cached)?;
                } else {
                    self.embedding_token_ids
                        .insert(rows, self.ops.i32_host_mirror(&device_values)?);
                }
                let image = self.prepared.as_ref().ok_or_else(|| Error::Internal {
                    message:
                        "CUDA transformer execution image was not compiled before embedding gather"
                            .into(),
                })?;
                let token_ids = self
                    .embedding_token_ids
                    .get(&rows)
                    .expect("embedding token buffer initialized above");
                self.ops.resident_embedding_hc_bf16_into(
                    image.embedding(),
                    token_ids.device(),
                    rows,
                    hc_mult,
                    output,
                )
            }
        }
        pub(crate) fn rms_norm_output_rows_device_into(
            &self,
            input: &cuda_linear::CudaF32Buffer,
            rows: usize,
            eps: f32,
            output: &mut cuda_linear::CudaF32Buffer,
        ) -> Result<()> {
            let image = self.prepared()?;
            let weight = image.output_norm();
            self.ops
                .rms_norm_rows_from_device_into(input, rows, weight, eps, output)
        }
        fn take_compact_i32_mirror(
            &mut self,
            compact_len: usize,
        ) -> Result<cuda_attention::CudaI32HostMirror> {
            if let Some(buffer) = self
                .compact_i32_mirrors
                .get_mut(&compact_len)
                .and_then(Vec::pop)
            {
                return Ok(buffer);
            }
            self.ops.i32_host_mirror(&vec![0; compact_len])
        }
        fn restore_compact_i32_mirror(
            &mut self,
            compact_len: usize,
            buffer: cuda_attention::CudaI32HostMirror,
        ) {
            self.compact_i32_mirrors
                .entry(compact_len)
                .or_default()
                .push(buffer);
        }
        fn arm_compact_download_completion(&mut self, pending: &mut CudaCompactI32Download) {
            if pending.callback_armed {
                return;
            }
            pending.callback_armed = self
                .ops
                .notify_control_stream(completion_notify_callback(self.completion_hub.clone()))
                .is_ok();
            if !pending.callback_armed {
                self.completion_hub.notify();
            }
        }
        fn poll_compact_i32_download(
            &mut self,
            pending: &mut CudaCompactI32Download,
        ) -> Result<Option<Vec<i32>>> {
            let mirror = pending.mirror.as_mut().ok_or_else(|| Error::Internal {
                message: "CUDA compact download mirror was already restored".into(),
            })?;
            let compact = self
                .ops
                .poll_i32_host_mirror_download(mirror, &pending.download)?;
            match compact_download_poll_action(compact.is_some(), pending.callback_armed) {
                CudaCompactDownloadPollAction::Consume => {}
                CudaCompactDownloadPollAction::Wait => return Ok(None),
                CudaCompactDownloadPollAction::ArmAndWait => {
                    self.arm_compact_download_completion(pending);
                    return Ok(None);
                }
            }
            Ok(compact)
        }
        fn restore_compact_i32_download(
            &mut self,
            pending: &mut CudaCompactI32Download,
        ) -> Result<()> {
            let mirror = pending.mirror.take().ok_or_else(|| Error::Internal {
                message: "CUDA compact download mirror was already restored".into(),
            })?;
            self.restore_compact_i32_mirror(pending.compact_len, mirror);
            Ok(())
        }
        pub(crate) fn begin_output_head_topk_rows(
            &mut self,
            hidden: &cuda_linear::CudaF32Buffer,
            batch_rows: usize,
            top_k: usize,
        ) -> Result<CudaOutputHeadTopKDownload> {
            {
                let (vocab, hidden_width) = {
                    let image = self.prepared()?;
                    let cuda_linear::CudaArtifactLinearShape::Bf16Bytes {
                        out_features,
                        in_features,
                    } = image.output_head().shape()
                    else {
                        return Err(Error::Model {
                            message: format!(
                                "CUDA transformer resident output head requires BF16 storage, got {:?}",
                                image.output_head().shape()
                            ),
                        });
                    };
                    (out_features, in_features)
                };
                let expected_hidden =
                    batch_rows
                        .checked_mul(hidden_width)
                        .ok_or_else(|| Error::Model {
                            message: "CUDA transformer output-head batch input size overflow"
                                .into(),
                        })?;
                if hidden.len() != expected_hidden {
                    return Err(Error::Model {
                        message: format!(
                            "CUDA transformer output-head input mismatch: expected {batch_rows}x{hidden_width}={expected_hidden}, got {}",
                            hidden.len()
                        ),
                    });
                }
                if batch_rows == 0 || top_k == 0 || top_k > vocab || top_k > 40 {
                    return Err(Error::Model {
                        message: format!(
                            "CUDA transformer output-head top-k requires rows>0 and k in 1..={}, got rows={batch_rows} k={top_k}",
                            vocab.min(40)
                        ),
                    });
                }
                if vocab > i32::MAX as usize {
                    return Err(Error::Model {
                        message: format!(
                            "CUDA transformer output-head vocab {vocab} exceeds the device i32 token-id limit"
                        ),
                    });
                }
                let logits_key = (batch_rows, vocab);
                if !self.output_head_logits.contains_key(&logits_key) {
                    let logits_len = batch_rows.checked_mul(vocab).ok_or_else(|| Error::Model {
                        message: "CUDA transformer full-vocab logits workspace overflow".into(),
                    })?;
                    self.output_head_logits
                        .insert(logits_key, self.ops.zero_f32_buffer(logits_len)?);
                }
                let output_len = batch_rows.checked_mul(top_k).ok_or_else(|| Error::Model {
                    message: "CUDA transformer output-head top-k workspace overflow".into(),
                })?;
                if !self.output_head_indices.contains_key(&output_len) {
                    self.output_head_indices
                        .insert(output_len, self.ops.zero_i32_buffer(output_len)?);
                    self.output_head_values
                        .insert(output_len, self.ops.zero_f32_buffer(output_len)?);
                }
                if !self
                    .output_head_linear_workspaces
                    .contains_key(&hidden.len())
                {
                    self.output_head_linear_workspaces.insert(
                        hidden.len(),
                        self.ops
                            .artifact_linear_workspace(batch_rows, hidden_width)?,
                    );
                }
                let compact_len = output_len.checked_mul(2).ok_or_else(|| Error::Model {
                    message: "CUDA transformer compact output-head workspace overflow".into(),
                })?;
                let mut mirror = self.take_compact_i32_mirror(compact_len)?;
                self.metrics.output_head_calls = self.metrics.output_head_calls.saturating_add(1);
                self.metrics.output_head_rows = self
                    .metrics
                    .output_head_rows
                    .saturating_add(batch_rows as u64);
                let topk_start = profile_start(self.profile);
                let image = self.prepared.as_ref().ok_or_else(|| Error::Internal {
                    message: "CUDA transformer execution image was not compiled before output-head execution".into(),
                })?;
                let logits = self
                    .output_head_logits
                    .get_mut(&logits_key)
                    .expect("output-head logits workspace initialized above");
                let indices = self
                    .output_head_indices
                    .get_mut(&output_len)
                    .expect("output-head indices workspace initialized above");
                let values = self
                    .output_head_values
                    .get_mut(&output_len)
                    .expect("output-head values workspace initialized above");
                let linear_workspace = self
                    .output_head_linear_workspaces
                    .get_mut(&hidden.len())
                    .expect("output-head linear workspace initialized above");
                self.ops
                    .artifact_linear_rows_from_device_into_with_scratch(
                        image.output_head(),
                        hidden,
                        batch_rows,
                        logits,
                        linear_workspace,
                    )?;
                self.ops.topk_vocab_rows_from_device_into(
                    logits, batch_rows, vocab, top_k, indices, values,
                )?;
                self.ops.pack_i32_f32_pairs_into(
                    indices,
                    values,
                    mirror.device_mut_invalidate_host(),
                    output_len,
                )?;
                let produced = self.ops.record_compute_event()?;
                let download = self
                    .ops
                    .begin_i32_host_mirror_download_after(&mut mirror, &produced)?;
                let mut compact = CudaCompactI32Download {
                    compact_len,
                    mirror: Some(mirror),
                    download,
                    callback_armed: false,
                };
                self.arm_compact_download_completion(&mut compact);
                Ok(CudaOutputHeadTopKDownload {
                    rows: batch_rows,
                    top_k,
                    vocab,
                    compact,
                    topk_start,
                })
            }
        }
        pub(crate) fn poll_output_head_topk_rows(
            &mut self,
            pending: &mut CudaOutputHeadTopKDownload,
        ) -> Result<Option<Vec<Vec<TokenLogit>>>> {
            let Some(compact) = self.poll_compact_i32_download(&mut pending.compact)? else {
                return Ok(None);
            };
            self.restore_compact_i32_download(&mut pending.compact)?;
            record_profile_duration(
                &mut self.metrics.output_head_topk_us,
                pending.topk_start.take(),
            );
            decode_output_head_topk_rows(&compact, pending.rows, pending.top_k, pending.vocab)
                .map(Some)
        }
        pub(crate) fn poll_output_head_topk_cancel_ready(
            &mut self,
            pending: &mut CudaOutputHeadTopKDownload,
        ) -> Result<bool> {
            if pending.compact.mirror.is_none() {
                return Ok(true);
            }
            if self
                .poll_compact_i32_download(&mut pending.compact)?
                .is_none()
            {
                return Ok(false);
            }
            self.restore_compact_i32_download(&mut pending.compact)?;
            Ok(true)
        }
    }
}
