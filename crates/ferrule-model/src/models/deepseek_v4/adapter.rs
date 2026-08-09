//! DeepSeek-V4 artifact adapter and generic decoder composition.

use std::path::Path;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use ferrule_common::CompletionHub;
#[cfg(feature = "cuda")]
use ferrule_common::execution::{
    ExecutionCapabilities, ExecutionTransactionId, KvBindingMode, LogitsRowPolicy,
};
use ferrule_common::{Error, ResidencyLeaseSet, Result};

use crate::ModelExecutionBackend;
use crate::decoder::GenericDecoderObservabilitySnapshot;
use crate::moe::prediction::{ExpertHotsetPredictor, ScoreBasedExpertPredictor};
#[cfg(feature = "cuda")]
use crate::moe::streaming::ExpertSourceCatalog;
#[cfg(feature = "cuda")]
use crate::moe::{RoutedMoePendingExpert, RoutedMoeQuiescence, RoutedMoeSequenceEvent};
#[cfg(feature = "cuda")]
use crate::runner::NativeProposal;
#[cfg(feature = "cuda")]
use crate::transformer::proposal::{
    MtpAttachment, MtpPhysicalOps, MtpProposalContinuation, MtpProposalExecutor, MtpProtocol,
    MtpStageResume, MtpStageStart,
};
#[cfg(feature = "cuda")]
use ferrule_backend::cuda::operators::{attention as cuda_attention, linear as cuda_linear};

use crate::decoder::{
    DecoderComponents, DecoderComposition, DecoderKvBackend, DecoderObserver,
    DecoderProvisionalRetainContext, DecoderResourceSnapshot, DecoderSequenceAttachment,
    DecoderTransactionContext, DecoderWait, GenericDecoderRunner, PagedKvBackend,
};
#[cfg(feature = "cuda")]
use crate::decoder::{DecoderLogits, DecoderTopKRow, NoTerminalGuard, PackedDecoderBatch};

#[cfg(feature = "cuda")]
use super::checkpoint::DeepSeekV4Checkpoint;
use super::recipe::{
    DeepSeekV4OutputProfileStats, DeepSeekV4PrepareOptions, DeepSeekV4PrepareProfile,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4LayerProfileStats {
    pub layer: usize,
    pub state_init_calls: u64,
    pub state_init_us: u64,
    pub decode_calls: u64,
    pub decode_total_us: u64,
    pub prefill_calls: u64,
    pub prefill_tokens: u64,
    pub prefill_total_us: u64,
    pub attn_hc_pre_us: u64,
    pub attn_norm_us: u64,
    pub attention_us: u64,
    pub attn_hc_post_us: u64,
    pub ffn_hc_pre_us: u64,
    pub ffn_norm_us: u64,
    pub moe_us: u64,
    pub ffn_hc_post_us: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4AttentionProfileStats {
    pub layer: usize,
    pub calls: u64,
    pub tokens: u64,
    pub q_a_us: u64,
    pub q_norm_us: u64,
    pub q_b_us: u64,
    pub q_head_norm_us: u64,
    pub q_rope_us: u64,
    pub kv_proj_us: u64,
    pub kv_norm_us: u64,
    pub kv_rope_quant_us: u64,
    pub kv_cache_append_us: u64,
    pub indexer_compress_us: u64,
    pub main_compress_us: u64,
    pub compressed_kv_upload_us: u64,
    pub topk_build_us: u64,
    pub sparse_attention_us: u64,
    pub context_rope_us: u64,
    pub output_a_us: u64,
    pub output_b_us: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4OperatorRuntimeCounters {
    pub kernel_launches: u64,
    pub host_to_device_copies: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_copies: u64,
    pub device_to_host_bytes: u64,
    pub artifact_uploads: u64,
    pub artifact_upload_bytes: u64,
    pub device_allocation_attempts: u64,
    pub device_allocations: u64,
    pub device_allocation_failures: u64,
    pub device_allocation_bytes: u64,
    pub stream_wide_syncs: u64,
    pub stream_wide_sync_failures: u64,
    pub moe_calls: u64,
    pub moe_total_us: u64,
    pub moe_input_prepare_us: u64,
    pub moe_gate_up_us: u64,
    pub moe_swiglu_us: u64,
    pub moe_down_us: u64,
    pub moe_routing_us: u64,
    pub moe_plan_us: u64,
    pub moe_workspace_us: u64,
    pub moe_commit_us: u64,
    pub output_head_calls: u64,
    pub output_head_rows: u64,
    pub output_head_topk_us: u64,
    pub expert_selected_load_requests: u64,
    pub expert_io_submitted_extents: u64,
    pub expert_io_completed_extents: u64,
    pub expert_io_failed_extents: u64,
    pub expert_io_requested_bytes: u64,
    pub expert_io_aligned_bytes: u64,
    pub expert_io_coalesced_slices: u64,
    pub expert_io_fixed_file_registrations: u64,
    pub expert_io_slab_exhaustions: u64,
    pub expert_io_peak_queue_depth: usize,
    pub expert_io_read_us: u64,
    pub arena_hits: u64,
    pub arena_misses: u64,
    pub arena_grows: u64,
    pub arena_reuses: u64,
    pub expert_residency_stats: ferrule_common::ExpertResidencyStats,
    pub expert_predictor_stats: crate::moe::ExpertPredictionStats,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4LoadProfile {
    pub checkpoint_us: u64,
    pub prepare: DeepSeekV4PrepareProfile,
    pub cuda_context_us: u64,
    pub sequence_state_us: u64,
    pub reader_us: u64,
    pub static_image_globals_us: u64,
    pub static_image_embedding_us: u64,
    pub static_image_output_head_us: u64,
    pub static_image_target_layers_us: u64,
    pub static_image_attachment_us: u64,
    pub static_image_total_us: u64,
    pub residency_us: u64,
    pub total_us: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeepSeekV4ObservabilitySnapshot {
    pub decoder: GenericDecoderObservabilitySnapshot,
    pub position: usize,
    pub load: DeepSeekV4LoadProfile,
    pub operator: DeepSeekV4OperatorRuntimeCounters,
    pub layers: Vec<DeepSeekV4LayerProfileStats>,
    pub attention: Vec<DeepSeekV4AttentionProfileStats>,
    pub output: DeepSeekV4OutputProfileStats,
    pub layer_runtime: Vec<DeepSeekV4LayerRuntimeStats>,
}

#[cfg(feature = "cuda")]
use std::convert::Infallible;

#[cfg(feature = "cuda")]
use super::recipe::{
    DeepSeekV4ExecutionPolicy, DeepSeekV4ModelPlan, DeepSeekV4Resources, prepare_with_policy,
};
#[cfg(feature = "cuda")]
use crate::moe::{ExpertLayerSources, RoutedMoePrefetchLayer, RoutedMoeResourceManager};
#[cfg(feature = "cuda")]
use crate::transformer::attention::mla::{
    MlaKvView, MlaLayerState, MlaPagedKvBinding, MlaPhysicalPool, MlaProposalAttentionBuffers,
    MlaProposalMainBuffers, MlaSequenceLifecycle,
};
use crate::transformer::attention::mla::{MlaSequenceAttachment, MlaSequenceState};
#[cfg(feature = "cuda")]
use crate::transformer::cuda::{
    CudaMlaHyperMoeArena, CudaMlaHyperMoeArenaVariants, CudaMlaHyperMoeContinuation,
    CudaMlaHyperMoeProgress, CudaMlaLayerRequest, CudaMtpHeadBuffers, CudaOutputHeadTopKDownload,
    CudaTransformerArenaKey, CudaTransformerBuffers, CudaTransformerRowLayout,
    CudaTransformerRuntimeHandle, MlaHyperMoeLayer,
};
#[cfg(feature = "cuda")]
use crate::transformer::{
    LayerRequest, Pending, Poll, Step, TransformerContinuation, TransformerForwardExecutor,
    TransformerModule,
};

#[derive(Debug, Clone)]
pub struct DeepSeekSequenceAttachment {
    pub(super) predictor: ScoreBasedExpertPredictor,
}

impl DeepSeekSequenceAttachment {
    fn new(num_layers: usize, num_experts: usize) -> Self {
        Self {
            predictor: ScoreBasedExpertPredictor::new(num_layers, num_experts),
        }
    }
}

impl DecoderSequenceAttachment for DeepSeekSequenceAttachment {
    type Release = ();

    fn preflight_release(&self) -> Result<Self::Release> {
        Ok(())
    }

    fn release(self, _release: Self::Release) {}
}

impl MlaSequenceAttachment for DeepSeekSequenceAttachment {
    fn reset_mla_sequence(&mut self) {
        self.predictor.clear();
    }
}

pub(super) type DeepSeekSequenceState =
    crate::decoder::DecoderSequenceState<DeepSeekSequenceAttachment, MlaSequenceState>;
#[cfg(feature = "cuda")]
type DeepSeekSequenceLifecycle = MlaSequenceLifecycle<DeepSeekSequenceAttachment>;

#[cfg(feature = "cuda")]
pub struct DeepSeekMtpArena {
    layers: Vec<CudaMlaHyperMoeArena>,
    hc_state: cuda_linear::CudaF32Buffer,
    attention: MlaProposalAttentionBuffers,
    head: CudaMtpHeadBuffers,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeepSeekMtpArena {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeepSeekMtpArena")
            .field("layers", &self.layers.len())
            .field("hc_state", &self.hc_state.len())
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
pub struct DeepSeekMtpStageContinuation {
    inner: Option<CudaMlaHyperMoeContinuation>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeepSeekMtpStageContinuation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeepSeekMtpStageContinuation")
            .field("active", &self.inner.is_some())
            .finish()
    }
}

#[cfg(feature = "cuda")]
pub struct DeepSeekMtpHeadDownload {
    inner: cuda_attention::CudaI32HostDownload,
    callback_armed: bool,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeepSeekMtpHeadDownload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeepSeekMtpHeadDownload")
            .field("callback_armed", &self.callback_armed)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
pub struct DeepSeekMoeQuiescence {
    inner: RoutedMoeQuiescence,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeepSeekMoeQuiescence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeepSeekMoeQuiescence")
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
pub struct DeepSeekMtpPhysicalOps {
    resources: Arc<DeepSeekV4Resources>,
    runtime: CudaTransformerRuntimeHandle,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeepSeekMtpPhysicalOps {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeepSeekMtpPhysicalOps")
            .field("stages", &self.stage_count())
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
impl DeepSeekMtpPhysicalOps {
    fn from_attachment(
        resources: Arc<DeepSeekV4Resources>,
        runtime: CudaTransformerRuntimeHandle,
    ) -> Result<Self> {
        let attachment = resources
            .proposal_attachment()
            .ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 Proposal physical executor requires an attachment".into(),
            })?;
        attachment.protocol()?;
        if attachment.stages.is_empty() || attachment.heads.is_none() {
            return Err(Error::Model {
                message: "DeepSeek-V4 Proposal attachment is incomplete".into(),
            });
        }
        Ok(Self { resources, runtime })
    }

    fn attachment(&self) -> &MtpAttachment<MlaHyperMoeLayer, Arc<ExpertSourceCatalog>> {
        self.resources
            .proposal_attachment()
            .expect("Proposal context was constructed from an attachment")
    }
}

#[cfg(feature = "cuda")]
impl MtpPhysicalOps for DeepSeekMtpPhysicalOps {
    type State = DeepSeekSequenceState;
    type KvView = MlaKvView;
    type Backend = PagedKvBackend<MlaPhysicalPool<DeepSeekSequenceAttachment>>;
    type Arena = DeepSeekMtpArena;
    type StageContinuation = DeepSeekMtpStageContinuation;
    type HeadDownload = DeepSeekMtpHeadDownload;
    type Quiescence = DeepSeekMoeQuiescence;

    fn protocol(&self) -> Result<MtpProtocol> {
        self.attachment().protocol()
    }

    fn validate_state(
        &self,
        _transaction: ExecutionTransactionId,
        state: &DeepSeekSequenceState,
    ) -> Result<()> {
        if state.kv_state().proposal_stage_count() != self.stage_count()
            || state.kv_state().paged_kv_binding().is_none()
        {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 Proposal state is not attachment-wired: stages={}/{} paged_kv={}",
                    state.kv_state().proposal_stage_count(),
                    self.stage_count(),
                    state.kv_state().paged_kv_binding().is_some()
                ),
            });
        }
        Ok(())
    }

    fn stage_count(&self) -> usize {
        self.attachment().stages.len()
    }

    fn acquire_arena(&mut self) -> Result<DeepSeekMtpArena> {
        let rows = cuda_linear::PROPOSAL_ROWS;
        let mut runtime = self.runtime.borrow_mut();
        runtime.check_arena_acquire()?;
        let mut layers = Vec::with_capacity(self.stage_count());
        for stage in &self.attachment().stages {
            layers.push(CudaMlaHyperMoeArena::new(
                &stage.backbone,
                rows,
                true,
                false,
                &mut runtime,
            )?);
        }
        let hc_len = rows
            .checked_mul(self.resources.config().hc_config().hc_hidden_size())
            .ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 Proposal HC arena size overflow".into(),
            })?;
        let hc_state = runtime.ops.zero_f32_buffer(hc_len)?;
        let attention = runtime.mla.allocate_proposal_attention_buffers()?;
        let head = runtime.allocate_mtp_head_buffers()?;
        Ok(DeepSeekMtpArena {
            layers,
            hc_state,
            attention,
            head,
        })
    }

    fn initialize(
        &mut self,
        transaction: ExecutionTransactionId,
        state: &mut DeepSeekSequenceState,
        _kv: &mut MlaKvView,
        anchor_token_id: u32,
        token_ids: &[u32],
        arena: &mut DeepSeekMtpArena,
    ) -> Result<()> {
        self.validate_state(transaction, state)?;
        if token_ids
            != self
                .attachment()
                .protocol()?
                .draft_input_ids(anchor_token_id)
        {
            return Err(Error::Model {
                message: "DeepSeek-V4 Proposal input rows do not match the attachment protocol"
                    .into(),
            });
        }
        self.runtime
            .borrow()
            .mtp_input_device_into(anchor_token_id, &mut arena.hc_state)
    }

    fn start_stage(
        &mut self,
        transaction: ExecutionTransactionId,
        state: &mut DeepSeekSequenceState,
        kv: &mut MlaKvView,
        stage: usize,
        sequence_tokens: usize,
        token_ids: &[u32],
        arena: &mut DeepSeekMtpArena,
    ) -> Result<MtpStageStart<DeepSeekMtpStageContinuation>> {
        self.validate_state(transaction, state)?;
        let transformer = &self
            .attachment()
            .stages
            .get(stage)
            .ok_or_else(|| Error::Model {
                message: format!("DeepSeek-V4 Proposal stage {stage} is unavailable"),
            })?
            .backbone;
        let progress = transformer.begin_device_hc_device(
            CudaMlaLayerRequest::Proposal {
                kv,
                stage,
                sequence_tokens,
                token_ids,
                attention: &mut arena.attention,
            },
            arena.layers.get_mut(stage).ok_or_else(|| Error::Internal {
                message: format!("DeepSeek-V4 Proposal arena stage {stage} is unavailable"),
            })?,
            &mut arena.hc_state,
            &mut self.runtime.borrow_mut(),
        )?;
        match progress {
            CudaMlaHyperMoeProgress::Waiting(inner) => {
                Ok(MtpStageStart::Waiting(DeepSeekMtpStageContinuation {
                    inner: Some(inner),
                }))
            }
            CudaMlaHyperMoeProgress::Complete { events } => {
                for event in events {
                    state.attachment_mut().predictor.observe_batch(event.event);
                }
                Ok(MtpStageStart::Complete)
            }
        }
    }

    fn resume_stage(
        &mut self,
        transaction: ExecutionTransactionId,
        state: &mut DeepSeekSequenceState,
        _kv: &mut MlaKvView,
        stage: usize,
        continuation: &mut DeepSeekMtpStageContinuation,
        lease: Option<&ResidencyLeaseSet>,
        arena: &mut DeepSeekMtpArena,
    ) -> Result<MtpStageResume> {
        self.validate_state(transaction, state)?;
        let transformer = &self
            .attachment()
            .stages
            .get(stage)
            .ok_or_else(|| Error::Model {
                message: format!("DeepSeek-V4 Proposal stage {stage} is unavailable"),
            })?
            .backbone;

        let inner = continuation.inner.take().ok_or_else(|| Error::Internal {
            message: "DeepSeek-V4 Proposal stage continuation was already consumed".into(),
        })?;
        match transformer.resume_device_hc_device(
            inner,
            lease,
            arena.layers.get_mut(stage).ok_or_else(|| Error::Internal {
                message: format!("DeepSeek-V4 Proposal arena stage {stage} is unavailable"),
            })?,
            &mut arena.hc_state,
            &mut self.runtime.borrow_mut(),
        )? {
            CudaMlaHyperMoeProgress::Waiting(inner) => {
                continuation.inner = Some(inner);
                Ok(MtpStageResume::Waiting)
            }
            CudaMlaHyperMoeProgress::Complete { events } => {
                for event in events {
                    state.attachment_mut().predictor.observe_batch(event.event);
                }
                Ok(MtpStageResume::Complete)
            }
        }
    }

    fn stage_needs_lease(&self, continuation: &Self::StageContinuation) -> bool {
        continuation
            .inner
            .as_ref()
            .is_some_and(|inner| !inner.pending_experts().is_empty())
    }

    fn stage_wait(
        &mut self,
        context: &DecoderTransactionContext,
        stage: usize,
        continuation: &DeepSeekMtpStageContinuation,
    ) -> Result<DecoderWait> {
        let pending = continuation
            .inner
            .as_ref()
            .ok_or_else(|| Error::Internal {
                message: "DeepSeek-V4 Proposal stage continuation was already consumed".into(),
            })?
            .pending_experts();
        let experts = self
            .resources
            .proposal_stage_experts()
            .get(stage)
            .ok_or_else(|| Error::Internal {
                message: format!("DeepSeek-V4 Proposal stage {stage} has no expert source catalog"),
            })?;
        expert_stage_wait(context, experts, pending)
    }

    fn submit_head(
        &mut self,
        transaction: ExecutionTransactionId,
        state: &mut DeepSeekSequenceState,
        _kv: &mut MlaKvView,
        anchor_token_id: u32,
        arena: &mut DeepSeekMtpArena,
    ) -> Result<DeepSeekMtpHeadDownload> {
        self.validate_state(transaction, state)?;
        let runtime = self.runtime.borrow();
        runtime.mtp_head_device_into(anchor_token_id, &arena.hc_state, &mut arena.head)?;
        let inner = runtime.begin_mtp_head_result_download(&mut arena.head)?;
        Ok(DeepSeekMtpHeadDownload {
            inner,
            callback_armed: false,
        })
    }

    fn poll_head(
        &mut self,
        transaction: ExecutionTransactionId,
        _kv: &mut MlaKvView,
        download: &mut DeepSeekMtpHeadDownload,
        arena: &mut DeepSeekMtpArena,
    ) -> Result<Option<NativeProposal>> {
        let _ = transaction;
        let runtime = self.runtime.borrow();
        let Some(compact) = runtime.poll_mtp_head_result(&mut arena.head, &download.inner)? else {
            download.callback_armed = false;
            return Ok(None);
        };
        let (token_ids, confidence_logits) = runtime.decode_mtp_head_result(compact)?;
        Ok(Some(NativeProposal {
            token_ids,
            confidence_logits,
        }))
    }

    fn arm_head_completion(
        &mut self,
        download: &mut DeepSeekMtpHeadDownload,
        completion: &ferrule_common::CompletionHub,
    ) {
        if download.callback_armed {
            return;
        }
        download.callback_armed = self
            .runtime
            .borrow()
            .ops
            .notify_control_stream(crate::runner::completion_notify_callback(
                completion.clone(),
            ))
            .is_ok();
        if !download.callback_armed {
            completion.notify();
        }
    }

    fn poll_stage_cancel_ready(
        &mut self,
        stage: usize,
        continuation: &mut DeepSeekMtpStageContinuation,
    ) -> Result<bool> {
        let backbone = &self
            .attachment()
            .stages
            .get(stage)
            .ok_or_else(|| Error::Model {
                message: format!("DeepSeek-V4 Proposal stage {stage} is unavailable"),
            })?
            .backbone;
        let inner = continuation.inner.as_mut().ok_or_else(|| Error::Internal {
            message: "DeepSeek-V4 Proposal stage continuation was already consumed".into(),
        })?;
        backbone.poll_cancel_ready(inner, &mut self.runtime.borrow_mut())
    }

    fn cancel_stage(
        &mut self,
        stage: usize,
        continuation: &mut DeepSeekMtpStageContinuation,
        _arena: &mut DeepSeekMtpArena,
    ) -> Result<()> {
        let inner = continuation.inner.take().ok_or_else(|| Error::Internal {
            message: "DeepSeek-V4 Proposal stage continuation was already consumed".into(),
        })?;
        self.attachment()
            .stages
            .get(stage)
            .ok_or_else(|| Error::Model {
                message: format!("DeepSeek-V4 Proposal stage {stage} is unavailable"),
            })?
            .backbone
            .cancel_device_hc_device(inner, &mut self.runtime.borrow_mut())
    }

    fn begin_quiescence(&mut self) -> Result<DeepSeekMoeQuiescence> {
        let mut runtime = self.runtime.borrow_mut();
        Ok(DeepSeekMoeQuiescence {
            inner: runtime.routed_moe_execution()?.begin_quiescence()?,
        })
    }

    fn poll_quiescence(&mut self, quiescence: &mut DeepSeekMoeQuiescence) -> Result<bool> {
        self.runtime
            .borrow_mut()
            .routed_moe_execution()?
            .poll_quiescence(&mut quiescence.inner)
    }

    fn retain_provisional(
        &mut self,
        context: &mut DecoderProvisionalRetainContext<
            '_,
            DeepSeekSequenceState,
            PagedKvBackend<MlaPhysicalPool<DeepSeekSequenceAttachment>>,
        >,
    ) -> Result<()> {
        context.with_retain_parts(
            |sources, working_states, executed_rows, retained_rows, backend, transaction| {
                backend.retain_provisional(
                    transaction,
                    sources,
                    working_states,
                    executed_rows,
                    retained_rows,
                )
            },
        )
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default)]
pub struct DeepSeekObserver;

#[cfg(feature = "cuda")]
impl DecoderObserver<crate::decoder::PagedKvBackend<MlaPhysicalPool<DeepSeekSequenceAttachment>>>
    for DeepSeekObserver
{
    type Snapshot = DeepSeekV4ObservabilitySnapshot;

    fn snapshot(
        &self,
        core: &GenericDecoderObservabilitySnapshot,
        backend: &crate::decoder::PagedKvBackend<MlaPhysicalPool<DeepSeekSequenceAttachment>>,
        _resources: &DecoderResourceSnapshot,
    ) -> Self::Snapshot {
        let capacity = backend.capacity();
        let mut decoder = core.clone();
        decoder.physical_kv_pages = capacity.physical_pages;
        decoder.resident_kv_pages = capacity.resident_pages;
        decoder.preempted_kv_pages = capacity.preempted_pages;
        decoder.free_kv_pages = capacity.free_pages;

        DeepSeekV4ObservabilitySnapshot {
            decoder,
            ..DeepSeekV4ObservabilitySnapshot::default()
        }
    }
}

/// Resolves the materialization wait edge for one layer's pending routed experts.
/// An empty pending set degrades to the generic completion edge armed by the
/// in-flight route download.
#[cfg(feature = "cuda")]
fn expert_stage_wait(
    context: &DecoderTransactionContext,
    experts: &ExpertLayerSources,
    pending: Vec<RoutedMoePendingExpert>,
) -> Result<DecoderWait> {
    if pending.is_empty() {
        return context.resolve_stage_wait(Vec::new(), crate::execution::WorkspaceClaim::NONE);
    }
    let placement = context.materialization_placement()?;
    let requests = pending
        .into_iter()
        .map(|operation| {
            let layer = u32::try_from(operation.layer).map_err(|_| Error::Execution {
                message: format!(
                    "DeepSeek-V4 expert layer {} exceeds the u32 ABI",
                    operation.layer
                ),
            })?;
            let expert = u32::try_from(operation.expert).map_err(|_| Error::Execution {
                message: format!(
                    "DeepSeek-V4 expert {} exceeds the u32 ABI",
                    operation.expert
                ),
            })?;
            let source = experts.source_catalog().require_resource_source(
                crate::moe::streaming::ExpertId::new(operation.layer, operation.expert),
            )?;
            let resource = ferrule_common::MaterializedResourceId::routed_expert(
                ferrule_common::LayerId::new(layer),
                ferrule_common::ExpertId::new(expert),
            );
            crate::execution::StageMaterializationRequest::new(
                crate::execution::StageResourceUse::new(
                    resource,
                    crate::execution::ResourceAccess::Read,
                    crate::execution::ResourceRetention::ThroughTransaction,
                ),
                crate::materialization::MaterializationRequest::for_placement(
                    placement, source, resource,
                )?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    context.resolve_stage_wait(requests, crate::execution::WorkspaceClaim::NONE)
}

/// Per-forward target-side custody. The rolling HC rows live in the arena;
/// per-sequence paged bindings are cloned once at embedding for layer requests.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DeepSeekHidden {
    paged_bindings: Vec<MlaPagedKvBinding>,
}

/// Shape-keyed forward arena: packed HC/output buffers, per-shape layer scratch,
/// and optional MTP main-projection buffers for target-tap capture.
#[cfg(feature = "cuda")]
pub struct DeepSeekForwardArena {
    buffers: CudaTransformerBuffers,
    layers: CudaMlaHyperMoeArenaVariants,
    proposal_main: Option<MlaProposalMainBuffers>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeepSeekForwardArena {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeepSeekForwardArena")
            .field("hc_rows", &self.buffers.hc_input.len())
            .field("proposal_main", &self.proposal_main.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
pub struct DeepSeekLayerPending {
    continuation: Option<CudaMlaHyperMoeContinuation>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeepSeekLayerPending {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeepSeekLayerPending")
            .field("active", &self.continuation.is_some())
            .finish()
    }
}

#[cfg(feature = "cuda")]
pub struct DeepSeekOutputPending {
    download: Option<CudaOutputHeadTopKDownload>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeepSeekOutputPending {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeepSeekOutputPending")
            .field("active", &self.download.is_some())
            .finish()
    }
}

#[cfg(feature = "cuda")]
pub struct DeepSeekModule {
    plan: DeepSeekV4ModelPlan,
    runtime: CudaTransformerRuntimeHandle,
}

#[cfg(feature = "cuda")]
impl DeepSeekModule {
    fn new(plan: DeepSeekV4ModelPlan, runtime: CudaTransformerRuntimeHandle) -> Self {
        Self { plan, runtime }
    }

    fn resources(&self) -> &DeepSeekV4Resources {
        self.plan.resources()
    }

    fn layer_exec(&self, layer: usize) -> Result<&MlaHyperMoeLayer> {
        self.resources()
            .layers()
            .get(layer)
            .ok_or_else(|| Error::Model {
                message: format!("DeepSeek-V4 target layer {layer} is unavailable"),
            })
    }
}

#[cfg(feature = "cuda")]
impl TransformerModule for DeepSeekModule {
    type State = DeepSeekSequenceState;
    type KvView = MlaKvView;
    type Hidden = DeepSeekHidden;
    type ArenaKey = CudaTransformerArenaKey;
    type Arena = DeepSeekForwardArena;
    type EmbeddingPending = Infallible;
    type LayerPending = DeepSeekLayerPending;
    type OutputPending = DeepSeekOutputPending;
    type Quiescence = DeepSeekMoeQuiescence;
    type Event = RoutedMoeSequenceEvent;
    type TerminalGuard = NoTerminalGuard;

    fn layer_count(&self) -> usize {
        self.resources().layers().len()
    }

    fn validate(
        &mut self,
        _context: &DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [DeepSeekSequenceState],
        _kv: &mut MlaKvView,
    ) -> Result<()> {
        let layers = self.layer_count();
        let proposal_stages = self
            .resources()
            .proposal_attachment()
            .map_or(0, |attachment| attachment.stages.len());
        for sequence in batch.sequences() {
            let state = states
                .get(sequence.state_index())
                .ok_or_else(|| Error::Model {
                    message: format!(
                        "DeepSeek-V4 forward state slot {} is unavailable",
                        sequence.state_index()
                    ),
                })?;
            let kv_state = state.kv_state();
            if kv_state.layers().len() != layers
                || kv_state.proposal_stage_count() != proposal_stages
            {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 forward state slot {} does not match the prepared transformer",
                        sequence.state_index()
                    ),
                });
            }
            if kv_state.paged_kv_binding().is_none() {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 forward state slot {} has no paged KV binding",
                        sequence.state_index()
                    ),
                });
            }
        }
        Ok(())
    }

    fn arena_key(&self, batch: &PackedDecoderBatch) -> Result<CudaTransformerArenaKey> {
        Ok(CudaTransformerArenaKey::new(
            batch.execution_shape_key()?,
            CudaTransformerRowLayout::IndependentRows,
        ))
    }

    fn build_arena(&mut self, key: &CudaTransformerArenaKey) -> Result<DeepSeekForwardArena> {
        let rows = key.shape().batch_tokens();
        let config = self.resources().config();
        let hidden_len = rows
            .checked_mul(config.hidden_size)
            .ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 forward hidden arena size overflow".into(),
            })?;
        let hc_len = rows
            .checked_mul(config.hc_config().hc_hidden_size())
            .ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 forward HC arena size overflow".into(),
            })?;
        let mut runtime = self.runtime.borrow_mut();
        runtime.check_arena_acquire()?;
        let buffers = runtime.allocate_decode_buffers(hc_len, hidden_len)?;
        let layers = CudaMlaHyperMoeArenaVariants::try_build_for_packed_mode(
            self.resources().layers(),
            rows,
            &mut runtime,
        )?;
        let proposal_main = match self.resources().proposal_attachment() {
            Some(attachment)
                if !attachment.target_layer_ids().is_empty()
                    && attachment
                        .target_layer_ids()
                        .iter()
                        .all(|layer| *layer < self.resources().layers().len()) =>
            {
                Some(runtime.allocate_proposal_main_buffers(rows)?)
            }
            _ => None,
        };
        Ok(DeepSeekForwardArena {
            buffers,
            layers,
            proposal_main,
        })
    }

    fn submit_embedding(
        &mut self,
        _context: &DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [DeepSeekSequenceState],
        _kv: &mut MlaKvView,
        arena: &mut DeepSeekForwardArena,
    ) -> Result<Step<DeepSeekHidden, Infallible>> {
        self.runtime
            .borrow_mut()
            .resident_embedding_hc_rows_into(batch.token_ids(), &mut arena.buffers.hc_input)?;
        let mut paged_bindings = Vec::with_capacity(batch.sequences().len());
        for sequence in batch.sequences() {
            let state = states
                .get(sequence.state_index())
                .ok_or_else(|| Error::Internal {
                    message: format!(
                        "DeepSeek-V4 embedding lost state slot {}",
                        sequence.state_index()
                    ),
                })?;
            paged_bindings.push(
                state
                    .kv_state()
                    .paged_kv_binding()
                    .cloned()
                    .ok_or_else(|| Error::Model {
                        message: format!(
                            "DeepSeek-V4 forward state slot {} has no paged KV binding",
                            sequence.state_index()
                        ),
                    })?,
            );
        }
        Ok(Step::Complete(DeepSeekHidden { paged_bindings }))
    }

    fn poll_embedding(
        &mut self,
        _context: &DecoderTransactionContext,
        _batch: &PackedDecoderBatch,
        _states: &mut [DeepSeekSequenceState],
        _kv: &mut MlaKvView,
        _arena: &mut DeepSeekForwardArena,
        pending: &mut Infallible,
    ) -> Result<Poll<DeepSeekHidden, Infallible>> {
        match *pending {}
    }

    fn start_layer(
        &mut self,
        context: &DecoderTransactionContext,
        layer: usize,
        request: LayerRequest<'_, DeepSeekSequenceState, MlaKvView>,
        hidden: &mut DeepSeekHidden,
        arena: &mut DeepSeekForwardArena,
    ) -> Result<Step<Vec<RoutedMoeSequenceEvent>, DeepSeekLayerPending>> {
        let LayerRequest::Target { batch, states, kv } = request else {
            return Err(Error::Model {
                message: "DeepSeek-V4 target forward cannot drive proposal stages".into(),
            });
        };
        let mut available = states
            .iter_mut()
            .map(|state| state.kv_state_mut().layers_mut().get_mut(layer))
            .collect::<Vec<_>>();
        let mut layer_states: Vec<&mut MlaLayerState> = Vec::with_capacity(batch.sequences().len());
        for sequence in batch.sequences() {
            let slot = available
                .get_mut(sequence.state_index())
                .ok_or_else(|| Error::Model {
                    message: format!(
                        "DeepSeek-V4 forward state slot {} is unavailable",
                        sequence.state_index()
                    ),
                })?;
            layer_states.push(slot.take().ok_or_else(|| Error::Model {
                message: format!(
                    "DeepSeek-V4 state slot {} is referenced more than once or misses layer {layer}",
                    sequence.state_index()
                ),
            })?);
        }
        let sequence_phases = batch
            .sequences()
            .iter()
            .map(|sequence| sequence.phase())
            .collect::<Vec<_>>();
        let progress = self.layer_exec(layer)?.begin_device_hc_device(
            CudaMlaLayerRequest::Target {
                kv,
                states: &mut layer_states,
                row_to_sequence: batch.row_to_sequence(),
                sequence_major_rows: batch.sequence_major_rows(),
                sequence_phases: &sequence_phases,
                paged_bindings: &hidden.paged_bindings,
                token_ids: batch.token_ids(),
                positions: batch.positions(),
            },
            arena
                .layers
                .get_for_layer_mut(layer)
                .ok_or_else(|| Error::Internal {
                    message: format!("DeepSeek-V4 layer arena {layer} is unavailable"),
                })?,
            &mut arena.buffers.hc_input,
            &mut self.runtime.borrow_mut(),
        )?;
        match progress {
            CudaMlaHyperMoeProgress::Complete { events } => Ok(Step::Complete(events)),
            CudaMlaHyperMoeProgress::Waiting(continuation) => {
                let pending_experts = continuation.pending_experts();
                if pending_experts.is_empty() {
                    return Ok(Step::Pending(Pending::physical(DeepSeekLayerPending {
                        continuation: Some(continuation),
                    })));
                }
                let experts =
                    self.resources()
                        .layer_experts()
                        .get(layer)
                        .ok_or_else(|| Error::Internal {
                            message: format!(
                                "DeepSeek-V4 layer {layer} has no expert source catalog"
                            ),
                        })?;
                let wait = expert_stage_wait(context, experts, pending_experts)?;
                Ok(Step::Pending(Pending::waiting(
                    DeepSeekLayerPending {
                        continuation: Some(continuation),
                    },
                    wait,
                )))
            }
        }
    }

    fn poll_layer(
        &mut self,
        context: &DecoderTransactionContext,
        layer: usize,
        request: LayerRequest<'_, DeepSeekSequenceState, MlaKvView>,
        _hidden: &mut DeepSeekHidden,
        arena: &mut DeepSeekForwardArena,
        pending: &mut DeepSeekLayerPending,
    ) -> Result<Poll<Vec<RoutedMoeSequenceEvent>, DeepSeekLayerPending>> {
        let LayerRequest::Target { .. } = request else {
            return Err(Error::Model {
                message: "DeepSeek-V4 target forward cannot drive proposal stages".into(),
            });
        };
        let continuation = pending.continuation.take().ok_or_else(|| Error::Internal {
            message: "DeepSeek-V4 layer continuation was already consumed".into(),
        })?;
        let leases = if continuation.pending_experts().is_empty() {
            None
        } else {
            Some(context.resume_lease()?)
        };
        let progress = self.layer_exec(layer)?.resume_device_hc_device(
            continuation,
            leases,
            arena
                .layers
                .get_for_layer_mut(layer)
                .ok_or_else(|| Error::Internal {
                    message: format!("DeepSeek-V4 layer arena {layer} is unavailable"),
                })?,
            &mut arena.buffers.hc_input,
            &mut self.runtime.borrow_mut(),
        )?;
        match progress {
            CudaMlaHyperMoeProgress::Complete { events } => Ok(Poll::Ready(events)),
            CudaMlaHyperMoeProgress::Waiting(continuation) => {
                let pending_experts = continuation.pending_experts();
                let wait = if pending_experts.is_empty() {
                    context
                        .resolve_stage_wait(Vec::new(), crate::execution::WorkspaceClaim::NONE)?
                } else {
                    let experts = self.resources().layer_experts().get(layer).ok_or_else(|| {
                        Error::Internal {
                            message: format!(
                                "DeepSeek-V4 layer {layer} has no expert source catalog"
                            ),
                        }
                    })?;
                    expert_stage_wait(context, experts, pending_experts)?
                };
                Ok(Poll::Pending(Pending::waiting(
                    DeepSeekLayerPending {
                        continuation: Some(continuation),
                    },
                    wait,
                )))
            }
        }
    }

    fn post_layer(
        &mut self,
        layer: usize,
        batch: &PackedDecoderBatch,
        _hidden: &DeepSeekHidden,
        arena: &mut DeepSeekForwardArena,
    ) -> Result<()> {
        let Some(proposal) = arena.proposal_main.as_mut() else {
            return Ok(());
        };
        self.runtime.borrow().capture_mtp_target_tap_from_device(
            layer,
            &arena.buffers.hc_input,
            batch.len(),
            &mut proposal.target_taps,
        )?;
        Ok(())
    }

    fn submit_output(
        &mut self,
        context: &DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        _states: &mut [DeepSeekSequenceState],
        kv: &mut MlaKvView,
        _hidden: &DeepSeekHidden,
        arena: &mut DeepSeekForwardArena,
    ) -> Result<Step<DecoderLogits, DeepSeekOutputPending>> {
        let rows = batch.len();
        let mut runtime = self.runtime.borrow_mut();
        if let Some(proposal) = arena.proposal_main.as_mut() {
            runtime.mtp_main_project_norm_device_into(rows, proposal)?;
            let attachment =
                self.resources()
                    .proposal_attachment()
                    .ok_or_else(|| Error::Internal {
                        message: "DeepSeek-V4 Proposal buffers exist without an attachment".into(),
                    })?;
            let max_position =
                batch
                    .positions()
                    .iter()
                    .copied()
                    .max()
                    .ok_or_else(|| Error::Model {
                        message: "DeepSeek-V4 Proposal positions are empty".into(),
                    })?;
            for (stage, stage_spec) in attachment.stages.iter().enumerate() {
                runtime.mla.proposal_context_kv_stage_packed_device_into(
                    kv,
                    stage,
                    stage_spec.execution_layer,
                    stage_spec.backbone.attention.config(),
                    rows,
                    max_position,
                    proposal,
                )?;
            }
        }
        let max_top_k = batch.max_top_k();
        if max_top_k == 0 {
            return Ok(Step::Complete(DecoderLogits::None));
        }
        runtime.hc_head_output_rows_device_into(
            &arena.buffers.hc_input,
            rows,
            &mut arena.buffers.final_hidden,
        )?;
        runtime.rms_norm_output_rows_device_into(
            &arena.buffers.final_hidden,
            rows,
            self.resources().config().norm_eps,
            &mut arena.buffers.topk_row,
        )?;
        let download =
            runtime.begin_output_head_topk_rows(&arena.buffers.topk_row, rows, max_top_k)?;
        drop(runtime);
        let wait =
            context.resolve_stage_wait(Vec::new(), crate::execution::WorkspaceClaim::NONE)?;
        Ok(Step::Pending(Pending::waiting(
            DeepSeekOutputPending {
                download: Some(download),
            },
            wait,
        )))
    }

    fn poll_output(
        &mut self,
        context: &DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        _states: &mut [DeepSeekSequenceState],
        _kv: &mut MlaKvView,
        _hidden: &DeepSeekHidden,
        _arena: &mut DeepSeekForwardArena,
        pending: &mut DeepSeekOutputPending,
    ) -> Result<Poll<DecoderLogits, DeepSeekOutputPending>> {
        let mut download = pending.download.take().ok_or_else(|| Error::Internal {
            message: "DeepSeek-V4 output-head download was already consumed".into(),
        })?;
        let Some(rows) = self
            .runtime
            .borrow_mut()
            .poll_output_head_topk_rows(&mut download)?
        else {
            let wait =
                context.resolve_stage_wait(Vec::new(), crate::execution::WorkspaceClaim::NONE)?;
            return Ok(Poll::Pending(Pending::waiting(
                DeepSeekOutputPending {
                    download: Some(download),
                },
                wait,
            )));
        };
        let mut logits = Vec::with_capacity(batch.logits_plan().rows().len());
        for requested in batch.logits_plan().rows() {
            let input_row = requested.input_row();
            let candidates = rows.get(input_row).ok_or_else(|| Error::Model {
                message: format!(
                    "DeepSeek-V4 output-head row {input_row} is outside {} decoded rows",
                    rows.len()
                ),
            })?;
            logits.push(DecoderTopKRow::new(input_row, candidates.clone()));
        }
        Ok(Poll::Ready(DecoderLogits::TopKRows(logits)))
    }

    fn poll_embedding_cancel(&mut self, pending: &mut Infallible) -> Result<bool> {
        match *pending {}
    }

    fn poll_layer_cancel(
        &mut self,
        layer: usize,
        pending: &mut DeepSeekLayerPending,
    ) -> Result<bool> {
        let continuation = pending
            .continuation
            .as_mut()
            .ok_or_else(|| Error::Internal {
                message: "DeepSeek-V4 layer continuation was already consumed".into(),
            })?;
        self.layer_exec(layer)?
            .poll_cancel_ready(continuation, &mut self.runtime.borrow_mut())
    }

    fn poll_output_cancel(&mut self, pending: &mut DeepSeekOutputPending) -> Result<bool> {
        let download = pending.download.as_mut().ok_or_else(|| Error::Internal {
            message: "DeepSeek-V4 output-head download was already consumed".into(),
        })?;
        self.runtime
            .borrow_mut()
            .poll_output_head_topk_cancel_ready(download)
    }

    fn cancel_embedding(&mut self, pending: Infallible) -> Result<()> {
        match pending {}
    }

    fn cancel_layer(&mut self, layer: usize, pending: DeepSeekLayerPending) -> Result<()> {
        let continuation = pending.continuation.ok_or_else(|| Error::Internal {
            message: "DeepSeek-V4 layer continuation was already consumed".into(),
        })?;
        self.layer_exec(layer)?
            .cancel_device_hc_device(continuation, &mut self.runtime.borrow_mut())
    }

    fn cancel_output(&mut self, pending: DeepSeekOutputPending) -> Result<()> {
        if pending.download.is_none() {
            return Err(Error::Internal {
                message: "DeepSeek-V4 output-head download was already consumed".into(),
            });
        }
        Ok(())
    }

    fn begin_quiescence(&mut self) -> Result<DeepSeekMoeQuiescence> {
        Ok(DeepSeekMoeQuiescence {
            inner: self
                .runtime
                .borrow_mut()
                .routed_moe_execution()?
                .begin_quiescence()?,
        })
    }

    fn poll_quiescence(&mut self, quiescence: &mut DeepSeekMoeQuiescence) -> Result<bool> {
        self.runtime
            .borrow_mut()
            .routed_moe_execution()?
            .poll_quiescence(&mut quiescence.inner)
    }

    fn finish(
        &mut self,
        batch: &PackedDecoderBatch,
        states: &mut [DeepSeekSequenceState],
        _kv: &mut MlaKvView,
        _arena: &mut DeepSeekForwardArena,
        events: Vec<RoutedMoeSequenceEvent>,
    ) -> Result<NoTerminalGuard> {
        for event in events {
            let sequence =
                batch
                    .sequences()
                    .get(event.sequence_index)
                    .ok_or_else(|| Error::Internal {
                        message: format!(
                            "DeepSeek-V4 MoE event references missing sequence {}",
                            event.sequence_index
                        ),
                    })?;
            let state = states
                .get_mut(sequence.state_index())
                .ok_or_else(|| Error::Internal {
                    message: format!(
                        "DeepSeek-V4 MoE event references missing state slot {}",
                        sequence.state_index()
                    ),
                })?;
            state.attachment_mut().predictor.observe_batch(event.event);
        }
        Ok(NoTerminalGuard)
    }

    fn abort(
        &mut self,
        _batch: &PackedDecoderBatch,
        _states: &mut [DeepSeekSequenceState],
        _kv: &mut MlaKvView,
        _arena: &mut DeepSeekForwardArena,
    ) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.runtime.borrow_mut().shutdown()
    }
}

#[cfg(feature = "cuda")]
pub struct DeepSeekDecoderComposition;

#[cfg(feature = "cuda")]
impl DecoderComposition for DeepSeekDecoderComposition {
    type Resources = Arc<DeepSeekV4Resources>;
    type SequenceState = DeepSeekSequenceState;
    type KvBackend = crate::decoder::PagedKvBackend<MlaPhysicalPool<DeepSeekSequenceAttachment>>;
    type ForwardExecutor = TransformerForwardExecutor<DeepSeekModule>;
    type Proposal = MtpProposalExecutor<DeepSeekMtpPhysicalOps>;
    type SequenceLifecycle = DeepSeekSequenceLifecycle;
    type ResourceManager = RoutedMoeResourceManager;
    type Observer = DeepSeekObserver;
    type Snapshot = DeepSeekV4ObservabilitySnapshot;
    type Continuation = TransformerContinuation<DeepSeekModule>;
    type ProposalContinuation = MtpProposalContinuation<DeepSeekMtpPhysicalOps>;
    type TerminalGuard = <DeepSeekModule as TransformerModule>::TerminalGuard;
}

#[cfg(feature = "cuda")]
fn deepseek_capabilities() -> ExecutionCapabilities {
    let max_packed_rows = usize::try_from(u32::MAX).unwrap_or(usize::MAX);
    ExecutionCapabilities {
        max_batch_tokens: max_packed_rows,
        max_sequences: usize::MAX,
        max_prefill_query_tokens_per_sequence: max_packed_rows,
        max_decode_query_tokens_per_sequence: 1,
        max_top_k: std::num::NonZeroU32::new(40),
        supports_prefill: true,
        supports_decode: true,
        supports_mixed: true,
        full_logits_width: None,
        kv_binding_mode: KvBindingMode::Paged,
        logits_row_policy: LogitsRowPolicy::Any,
    }
}

/// DeepSeek-V4 artifact adapter and runner construction boundary.
pub struct DeepSeekV4Adapter;

impl DeepSeekV4Adapter {
    #[cfg(feature = "cuda")]
    pub fn new_with_operator_backend(
        model: DeepSeekV4Checkpoint,
        options: DeepSeekV4PrepareOptions,
        operator_backend: ModelExecutionBackend,
    ) -> Result<GenericDecoderRunner<DeepSeekDecoderComposition>> {
        if operator_backend != ModelExecutionBackend::Cuda {
            return Err(adapter_error(
                "resident inference requires the CUDA packed execution backend",
            ));
        }
        Self::build_cuda(model, options)
    }

    #[cfg(feature = "cuda")]
    pub fn load_hf_with_options(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: DeepSeekV4PrepareOptions,
    ) -> Result<GenericDecoderRunner<DeepSeekDecoderComposition>> {
        Self::load_hf_with_options_and_backend(
            model_dir,
            max_tensor_bytes,
            options,
            ModelExecutionBackend::Cuda,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn load_hf_with_options_and_backend(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: DeepSeekV4PrepareOptions,
        operator_backend: ModelExecutionBackend,
    ) -> Result<GenericDecoderRunner<DeepSeekDecoderComposition>> {
        Self::new_with_operator_backend(
            DeepSeekV4Checkpoint::load_hf_with_limit(model_dir, max_tensor_bytes)?,
            options,
            operator_backend,
        )
    }

    #[cfg(feature = "cuda")]
    fn build_cuda(
        model: DeepSeekV4Checkpoint,
        options: DeepSeekV4PrepareOptions,
    ) -> Result<GenericDecoderRunner<DeepSeekDecoderComposition>> {
        let completion_hub = CompletionHub::new();
        let policy = DeepSeekV4ExecutionPolicy::resolve()?;
        let plan = prepare_with_policy(&model, options, policy)?;
        let resources = Arc::clone(plan.resources());
        let model_info = resources.model_info();
        let page_size = resources.kv_layout().page_size();
        let prepared_plan_id = plan.generation();
        let (resource_manager, runtime, completion_reactors) =
            initialize_cuda_runtime(&resources, prepared_plan_id, completion_hub.clone())?;
        let cuda_operators = std::rc::Rc::clone(&runtime.ops);
        let proposal_stage_count = resources
            .proposal_attachment()
            .map_or(0, |attachment| attachment.stages.len());
        let sequence_lifecycle = DeepSeekSequenceLifecycle::new(
            DeepSeekSequenceAttachment::new(
                resources.config().num_layers + proposal_stage_count,
                resources.config().num_routed_experts,
            ),
            resources
                .layers()
                .iter()
                .map(|layer| layer.attention.config()),
            resources
                .proposal_attachment()
                .into_iter()
                .flat_map(|attachment| attachment.stages.iter())
                .map(|stage| stage.backbone.attention.config()),
            std::rc::Rc::clone(&cuda_operators),
        );
        let default_state = sequence_lifecycle.create_state()?;

        let backend = crate::decoder::PagedKvBackend::new(MlaPhysicalPool::<
            DeepSeekSequenceAttachment,
        >::new(
            cuda_operators,
            resources.kv_layout().clone(),
        ));
        let runtime = std::rc::Rc::new(std::cell::RefCell::new(runtime));
        let proposal = match resources.proposal_attachment() {
            Some(attachment) => MtpProposalExecutor::new(
                prepared_plan_id,
                attachment.protocol()?,
                DeepSeekMtpPhysicalOps::from_attachment(
                    Arc::clone(&resources),
                    std::rc::Rc::clone(&runtime),
                )?,
            )?,
            None => MtpProposalExecutor::target_only(),
        };
        let forward_executor = TransformerForwardExecutor::new(DeepSeekModule::new(plan, runtime));
        let tokenizer = model.into_tokenizer();

        GenericDecoderRunner::from_runtime(DecoderComponents {
            resources,
            model_info,
            tokenizer,
            capabilities: deepseek_capabilities(),
            page_size,
            prepared_plan_id,
            forward_executor,
            backend,
            default_state,
            sequence_lifecycle,
            proposal,
            resource_manager,
            observer: DeepSeekObserver,
            completion_hub,
            completion_reactors,
        })
    }
}

#[cfg(feature = "cuda")]
static NEXT_DEEPSEEK_MODEL_INSTANCE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

/// Constructs the CUDA transformer runtime and the routed-MoE resource
/// manager for one assembled DeepSeek plan.
#[cfg(feature = "cuda")]
fn initialize_cuda_runtime(
    resources: &DeepSeekV4Resources,
    generation: u64,
    completion_hub: CompletionHub,
) -> Result<(
    RoutedMoeResourceManager,
    crate::transformer::cuda::CudaTransformerRuntime,
    Vec<crate::runner::ModelCompletionReactor>,
)> {
    use crate::moe::RoutedMoeResidencyConfig;
    use crate::moe::streaming::{ExpertIoPlan, ExpertStreamingReader};
    use crate::transformer::cuda::{CudaTransformerRuntime, CudaTransformerRuntimeConfig};

    let policy = resources.policy();
    let runtime_config = CudaTransformerRuntimeConfig::new(
        policy.profile_enabled(),
        policy.profile_sync(),
        policy.managed_experts(),
    );
    let mut runtime = CudaTransformerRuntime::new_with_completion_hub(
        runtime_config,
        completion_hub.clone(),
        resources.prepared().clone(),
    )?;
    let io_plan = ExpertIoPlan::for_checkpoint_plans(
        resources
            .expert_materialization_sources()
            .iter()
            .map(|entry| entry.read_plan()),
    )?;
    let (reader, io_plan) = ExpertStreamingReader::from_env_with_cuda_pinned(
        resources.prepare_options().expert_reader_max_tensor_bytes,
        runtime.pinned_host_allocator(),
        completion_hub,
        io_plan,
    )?;
    let residency = RoutedMoeResidencyConfig {
        num_routed_experts: resources.config().num_routed_experts,
        num_experts_per_tok: resources.config().num_experts_per_tok,
        reserved_device_bytes: resources.prepare_options().reserved_device_bytes,
        hotset_experts: resources.prepare_options().moe_hotset_experts,
        expert_upload_inflight: policy.expert_upload_inflight(),
    };
    let io_limits = RoutedMoeResourceManager::physical_io_limits(
        resources.layer_experts(),
        resources.proposal_stage_experts(),
        &reader,
        io_plan,
        &residency,
    )?;
    let completion_reactors = reader.take_completion_reactors();

    runtime.compile(generation, resources.cuda_source()?)?;
    let (free_device_bytes, _) = runtime.memory_info()?;
    let free_device_bytes = u64::try_from(free_device_bytes)
        .map_err(|_| adapter_error("CUDA free-memory byte count exceeds u64"))?;
    let compute = runtime.compute_stream_authority();

    if resources.layers().len() != resources.layer_experts().len() {
        return Err(Error::Internal {
            message: "DeepSeek-V4 layer/expert catalog count mismatch".into(),
        });
    }
    let prefetch = resources
        .layers()
        .iter()
        .zip(resources.layer_experts())
        .map(|(layer, experts)| RoutedMoePrefetchLayer {
            layer: layer.layer,
            top_k: layer.feed_forward.router_policy.top_k,
            expert_limit: resources.config().num_routed_experts,
            hash: layer.feed_forward.router.hash_table.clone().map(Arc::from),
            rows: layer.feed_forward.router.hash_rows,
            cols: layer.feed_forward.router.hash_cols,
            sources: Arc::clone(experts.source_catalog()),
        })
        .collect::<Vec<_>>();
    let mut manager = RoutedMoeResourceManager::new(io_limits, prefetch)?;
    let subsystem = manager.initialize_cuda(
        resources.expert_materialization_sources(),
        residency,
        reader,
        NEXT_DEEPSEEK_MODEL_INSTANCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        free_device_bytes,
        compute,
    )?;
    runtime.configure_expert_subsystem(subsystem)?;
    Ok((manager, runtime, completion_reactors))
}

fn adapter_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("DeepSeek-V4 adapter: {}", message.into()),
    }
}
