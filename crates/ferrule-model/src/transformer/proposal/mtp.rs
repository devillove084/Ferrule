//! Model-independent multi-token prediction (MTP) attachment and executor.
//!
//! MTP owns the proposal row protocol, target-tap projection, Markov/confidence
//! head contract, and physical continuation state machine. Generic decoder
//! transaction, continuation-ID, resolver, lease, and runner custody remain in
//! `crate::decoder`.

use std::fmt::Debug;

use ferrule_backend::cpu::{bf16_rne, rms_norm};
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{CompletionHub, Error, ResidencyLeaseSet, Result};

use crate::checkpoint::weight::LinearWeight;
use crate::decoder::{
    DecoderCancelProgress, DecoderKvBackend, DecoderProposalExecutor, DecoderProposalProgress,
    DecoderProposalResumeProgress, DecoderProvisionalRetainContext, DecoderSequence,
    DecoderTransactionContext, DecoderWait,
};
use crate::runner::{NativeProposal, NativeProposalSource};
use crate::transformer::connection::HyperConnectionHead;

/// Stable implementation identity for the generic checkpoint-native MTP path.
pub const MTP_PROPOSAL_IMPLEMENTATION: &str = "mtp-physical-v1";

/// Model-independent MTP configuration parsed by a model recipe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpConfig {
    pub block_size: usize,
    pub noise_token_id: Option<u32>,
    pub target_layer_ids: Vec<usize>,
    pub markov_rank: Option<usize>,
}

impl MtpConfig {
    pub fn protocol(&self) -> Result<MtpProtocol> {
        MtpProtocol::try_from(self)
    }
}

/// Frozen MTP anchor/noise/bonus row contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpProtocol {
    /// Number of draft tokens produced by one native block.
    pub gamma: usize,
    /// Backbone rows: one anchor followed by noise rows.
    pub draft_backbone_rows: usize,
    /// Target rows: one carried anchor followed by all draft tokens.
    pub target_verify_rows: usize,
    /// Full acceptance may externally commit every draft plus the bonus row.
    pub max_external_commit_tokens: usize,
    pub noise_token_id: u32,
    pub target_layer_ids: Vec<usize>,
}

impl TryFrom<&MtpConfig> for MtpProtocol {
    type Error = Error;

    fn try_from(config: &MtpConfig) -> Result<Self> {
        if config.block_size == 0 {
            return Err(mtp_error("gamma must be greater than zero"));
        }
        let noise_token_id = config
            .noise_token_id
            .ok_or_else(|| mtp_error("protocol requires a noise token id"))?;
        if config.target_layer_ids.is_empty() {
            return Err(mtp_error("protocol requires target hidden-state layers"));
        }
        if config
            .target_layer_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(mtp_error("target layers must be strictly increasing"));
        }
        let target_verify_rows = config
            .block_size
            .checked_add(1)
            .ok_or_else(|| mtp_error("target width overflow"))?;
        Ok(Self {
            gamma: config.block_size,
            draft_backbone_rows: config.block_size,
            target_verify_rows,
            max_external_commit_tokens: target_verify_rows,
            noise_token_id,
            target_layer_ids: config.target_layer_ids.clone(),
        })
    }
}

impl MtpProtocol {
    /// Builds `[anchor, noise × (gamma - 1)]` for the physical backbone.
    pub fn draft_input_ids(&self, anchor_token_id: u32) -> Vec<u32> {
        let mut input = vec![self.noise_token_id; self.draft_backbone_rows];
        input[0] = anchor_token_id;
        input
    }

    /// Includes the carried anchor/bonus target row in verification width.
    pub fn target_rows_for_drafts(&self, proposed_draft_tokens: usize) -> Result<usize> {
        if proposed_draft_tokens > self.gamma {
            return Err(mtp_error(format!(
                "requested {proposed_draft_tokens} drafts above gamma {}",
                self.gamma
            )));
        }
        proposed_draft_tokens
            .checked_add(1)
            .ok_or_else(|| mtp_error("target width overflow"))
    }
}

/// One MTP transformer stage. Model-specific backbone and expert payloads remain typed.
pub struct MtpStage<B, E> {
    pub index: usize,
    pub execution_layer: usize,
    pub backbone: B,
    pub main_projection: Option<LinearWeight>,
    pub main_norm: Option<Vec<f32>>,
    pub experts: E,
}

/// Final MTP Markov and confidence heads.
pub struct MtpHeads {
    pub hyper_connection_head: HyperConnectionHead,
    pub norm: Vec<f32>,
    /// Token embedding into the low-rank Markov state, `[vocab, rank]`.
    pub markov_embedding: LinearWeight,
    /// Low-rank Markov state projected to vocabulary bias, `[vocab, rank]`.
    pub markov_output: LinearWeight,
    /// Confidence over concatenated `[hidden, markov]`, `[1, hidden + rank]`.
    pub confidence: LinearWeight,
}

/// One reference head result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtpHeadOutput {
    pub token_id: u32,
    pub confidence_logit: f32,
}

impl MtpHeads {
    pub fn markov_rank(&self) -> usize {
        self.markov_embedding.format.in_features()
    }

    pub fn validate(&self, hidden_size: usize, vocab_size: usize) -> Result<()> {
        let rank = self.markov_rank();
        let confidence_width = hidden_size
            .checked_add(rank)
            .ok_or_else(|| mtp_error("confidence width overflow"))?;
        if rank == 0
            || self.markov_embedding.format.out_features() != vocab_size
            || self.markov_output.format.in_features() != rank
            || self.markov_output.format.out_features() != vocab_size
            || self.confidence.format.in_features() != confidence_width
            || self.confidence.format.out_features() != 1
            || self.norm.len() != hidden_size
        {
            return Err(mtp_error(
                "Markov/confidence head dimensions are inconsistent",
            ));
        }
        Ok(())
    }

    /// Reference head math: `argmax(lm(hidden) + W2 * W1[token])` and
    /// `confidence([hidden, W1[token]])`.
    pub fn reference_step(
        &self,
        previous_token_id: u32,
        hidden: &[f32],
        base_logits: &[f32],
    ) -> Result<MtpHeadOutput> {
        let vocab = self.markov_embedding.format.out_features();
        let hidden_size = hidden.len();
        self.validate(hidden_size, vocab)?;
        let previous = usize::try_from(previous_token_id)
            .map_err(|_| mtp_error("previous token exceeds usize"))?;
        if previous >= vocab || base_logits.len() != vocab {
            return Err(mtp_error("head input shape or previous token is invalid"));
        }

        let rank = self.markov_rank();
        let embedding = self.markov_embedding.reference_weights_f32()?;
        let markov = &embedding[previous * rank..(previous + 1) * rank];
        let bias = self.markov_output.reference_matvec(markov)?;
        let token = base_logits
            .iter()
            .zip(&bias)
            .enumerate()
            .max_by(
                |(left_index, (left_base, left_bias)), (right_index, (right_base, right_bias))| {
                    (**left_base + **left_bias)
                        .total_cmp(&(**right_base + **right_bias))
                        .then_with(|| right_index.cmp(left_index))
                },
            )
            .map(|(token, _)| token)
            .ok_or_else(|| mtp_error("head vocabulary is empty"))?;

        let mut confidence_input = Vec::with_capacity(hidden_size + rank);
        confidence_input.extend_from_slice(hidden);
        confidence_input.extend_from_slice(markov);
        let confidence = self.confidence.reference_matvec(&confidence_input)?;
        Ok(MtpHeadOutput {
            token_id: u32::try_from(token).map_err(|_| mtp_error("token exceeds u32"))?,
            confidence_logit: confidence[0],
        })
    }
}

/// Complete model-independent MTP attachment.
pub struct MtpAttachment<B, E> {
    pub stages: Vec<MtpStage<B, E>>,
    pub heads: Option<MtpHeads>,
    pub config: MtpConfig,
    pub hidden_size: usize,
    pub norm_eps: f32,
}

impl<B, E> MtpAttachment<B, E> {
    pub const fn block_size(&self) -> usize {
        self.config.block_size
    }

    pub const fn noise_token_id(&self) -> Option<u32> {
        self.config.noise_token_id
    }

    pub fn target_layer_ids(&self) -> &[usize] {
        &self.config.target_layer_ids
    }

    pub const fn markov_rank(&self) -> Option<usize> {
        self.config.markov_rank
    }

    pub fn protocol(&self) -> Result<MtpProtocol> {
        self.config.protocol()
    }

    /// CPU oracle for stage-zero `main_norm(main_projection(target_taps))`.
    pub fn stage_zero_main_reference(&self, target_taps: &[f32], rows: usize) -> Result<Vec<f32>> {
        let stage_zero = self
            .stages
            .first()
            .ok_or_else(|| mtp_error("stage zero is missing"))?;
        let projection = stage_zero
            .main_projection
            .as_ref()
            .ok_or_else(|| mtp_error("stage-zero main projection is missing"))?;
        let norm = stage_zero
            .main_norm
            .as_deref()
            .ok_or_else(|| mtp_error("stage-zero main norm is missing"))?;
        let input_size = projection.format.in_features();
        let output_size = projection.format.out_features();
        let expected = rows
            .checked_mul(input_size)
            .ok_or_else(|| mtp_error("stage-zero input size overflow"))?;
        if rows == 0
            || target_taps.len() != expected
            || output_size != self.hidden_size
            || norm.len() != output_size
        {
            return Err(mtp_error("stage-zero target-tap shape mismatch"));
        }
        let mut output = Vec::with_capacity(rows * output_size);
        for row in 0..rows {
            let start = row * input_size;
            let mut taps = target_taps[start..start + input_size].to_vec();
            taps.iter_mut().for_each(|value| *value = bf16_rne(*value));
            let mut projected = projection.reference_matvec(&taps)?;
            projected
                .iter_mut()
                .for_each(|value| *value = bf16_rne(*value));
            let normalized = rms_norm(&projected, norm, self.norm_eps)?;
            output.extend(normalized.into_iter().map(bf16_rne));
        }
        Ok(output)
    }

    /// CPU oracle for the final HC reduction/norm and sequential Markov head.
    pub fn heads_reference(
        &self,
        hc_state: &[f32],
        anchor_token_id: u32,
        lm_head: &LinearWeight,
    ) -> Result<NativeProposal> {
        let heads = self
            .heads
            .as_ref()
            .ok_or_else(|| mtp_error("prediction heads are missing"))?;
        let rows = self.block_size();
        let mut hidden = heads.hyper_connection_head.reference(hc_state, rows)?;
        for row in hidden.chunks_exact_mut(self.hidden_size) {
            row.iter_mut().for_each(|value| *value = bf16_rne(*value));
            let normalized = rms_norm(row, &heads.norm, self.norm_eps)?;
            row.copy_from_slice(&normalized.into_iter().map(bf16_rne).collect::<Vec<_>>());
        }
        let mut previous = anchor_token_id;
        let mut token_ids = Vec::with_capacity(rows);
        let mut confidence_logits = Vec::with_capacity(rows);
        for row in hidden.chunks_exact(self.hidden_size) {
            let base_logits = lm_head.reference_matvec(row)?;
            let result = heads.reference_step(previous, row, &base_logits)?;
            token_ids.push(result.token_id);
            confidence_logits.push(result.confidence_logit);
            previous = result.token_id;
        }
        Ok(NativeProposal {
            token_ids,
            confidence_logits,
        })
    }
}

/// Typed prepared MTP stage payload stored in a `PreparedDecoder` attachment.
pub struct PreparedMtpStage<B, L, N> {
    pub execution_layer: usize,
    pub backbone: B,
    pub main_projection: Option<L>,
    pub main_norm: Option<N>,
}

/// Typed prepared Markov/confidence handles.
pub struct PreparedMtpHeads<H, N, L> {
    pub hyper_connection_head: H,
    pub norm: N,
    pub markov_embedding: L,
    pub markov_output: L,
    pub confidence: L,
}

/// Device-ready MTP payload with no model-local cache or resolver lifecycle.
pub struct PreparedMtpAttachment<B, H, N, L, P> {
    pub config: MtpConfig,
    pub stages: Box<[PreparedMtpStage<B, L, N>]>,
    pub heads: PreparedMtpHeads<H, N, L>,
    pub compiled_plan: P,
}

/// Result of starting one physical MTP stage.
#[derive(Debug)]
pub enum MtpStageStart<C> {
    Waiting(C),
    Complete,
}

/// Result of resuming one physical MTP stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpStageResume {
    Waiting,
    Complete,
}

/// Narrow typed boundary between the generic MTP phase machine and model/device ops.
///
/// This trait exists because the MTP executor and at least one model implementation
/// are distinct implementation points. Every physical value is an associated type;
/// no erased context or downcast is permitted.
pub trait MtpPhysicalOps: Debug {
    type State: DecoderSequence;
    type KvView;
    type Backend: DecoderKvBackend<SequenceState = Self::State, KvView = Self::KvView>;
    type Arena: Debug;
    type StageContinuation: Debug;
    type HeadDownload: Debug;
    type Quiescence: Debug;

    fn protocol(&self) -> Result<MtpProtocol>;
    fn stage_count(&self) -> usize;
    fn validate_state(
        &self,
        transaction: ExecutionTransactionId,
        state: &Self::State,
    ) -> Result<()>;
    fn acquire_arena(&mut self) -> Result<Self::Arena>;
    fn initialize(
        &mut self,
        transaction: ExecutionTransactionId,
        state: &mut Self::State,
        kv: &mut Self::KvView,
        anchor_token_id: u32,
        token_ids: &[u32],
        arena: &mut Self::Arena,
    ) -> Result<()>;
    #[allow(clippy::too_many_arguments)]
    fn start_stage(
        &mut self,
        transaction: ExecutionTransactionId,
        state: &mut Self::State,
        kv: &mut Self::KvView,
        stage: usize,
        sequence_tokens: usize,
        token_ids: &[u32],
        arena: &mut Self::Arena,
    ) -> Result<MtpStageStart<Self::StageContinuation>>;
    #[allow(clippy::too_many_arguments)]
    fn resume_stage(
        &mut self,
        transaction: ExecutionTransactionId,
        state: &mut Self::State,
        kv: &mut Self::KvView,
        stage: usize,
        continuation: &mut Self::StageContinuation,
        lease: Option<&ResidencyLeaseSet>,
        arena: &mut Self::Arena,
    ) -> Result<MtpStageResume>;
    fn stage_needs_lease(&self, continuation: &Self::StageContinuation) -> bool;
    fn stage_wait(
        &mut self,
        context: &DecoderTransactionContext,
        stage: usize,
        continuation: &Self::StageContinuation,
    ) -> Result<DecoderWait>;
    fn submit_head(
        &mut self,
        transaction: ExecutionTransactionId,
        state: &mut Self::State,
        kv: &mut Self::KvView,
        anchor_token_id: u32,
        arena: &mut Self::Arena,
    ) -> Result<Self::HeadDownload>;
    fn poll_head(
        &mut self,
        transaction: ExecutionTransactionId,
        kv: &mut Self::KvView,
        download: &mut Self::HeadDownload,
        arena: &mut Self::Arena,
    ) -> Result<Option<NativeProposal>>;
    fn arm_head_completion(
        &mut self,
        download: &mut Self::HeadDownload,
        completion: &CompletionHub,
    );
    fn poll_stage_cancel_ready(
        &mut self,
        stage: usize,
        continuation: &mut Self::StageContinuation,
    ) -> Result<bool>;
    fn cancel_stage(
        &mut self,
        stage: usize,
        continuation: &mut Self::StageContinuation,
        arena: &mut Self::Arena,
    ) -> Result<()>;
    fn begin_quiescence(&mut self) -> Result<Self::Quiescence>;
    fn poll_quiescence(&mut self, quiescence: &mut Self::Quiescence) -> Result<bool>;
    fn retain_provisional(
        &mut self,
        context: &mut DecoderProvisionalRetainContext<'_, Self::State, Self::Backend>,
    ) -> Result<()>;
}

/// Model-independent physical MTP continuation.
#[derive(Debug)]
pub struct MtpProposalContinuation<P>
where
    P: MtpPhysicalOps,
{
    anchor_token_id: u32,
    sequence_tokens: usize,
    arena: P::Arena,
    phase: MtpPhase<P>,
}

#[derive(Debug)]
enum MtpPhase<P>
where
    P: MtpPhysicalOps,
{
    Initializing,
    StageReady {
        stage: usize,
    },
    StageWaiting {
        stage: usize,
        continuation: P::StageContinuation,
    },
    HeadWaiting {
        download: P::HeadDownload,
    },
    Finalizing,
    Failed {
        owner: MtpPhysicalOwner<P>,
    },
    CancelPending {
        owner: MtpPhysicalOwner<P>,
    },
    Cancelling {
        owner: MtpPhysicalOwner<P>,
        quiescence: P::Quiescence,
    },
}

#[derive(Debug)]
enum MtpPhysicalOwner<P>
where
    P: MtpPhysicalOps,
{
    Arena,
    Stage {
        stage: usize,
        continuation: P::StageContinuation,
    },
    Head {
        download: P::HeadDownload,
    },
}

impl<P> MtpPhase<P>
where
    P: MtpPhysicalOps,
{
    fn into_owner(self) -> MtpPhysicalOwner<P> {
        match self {
            Self::Initializing | Self::StageReady { .. } | Self::Finalizing => {
                MtpPhysicalOwner::Arena
            }
            Self::StageWaiting {
                stage,
                continuation,
            } => MtpPhysicalOwner::Stage {
                stage,
                continuation,
            },
            Self::HeadWaiting { download } => MtpPhysicalOwner::Head { download },
            Self::Failed { owner }
            | Self::CancelPending { owner }
            | Self::Cancelling { owner, .. } => owner,
        }
    }
}

impl<P> MtpProposalContinuation<P>
where
    P: MtpPhysicalOps,
{
    fn take_phase(&mut self) -> MtpPhase<P> {
        std::mem::replace(
            &mut self.phase,
            MtpPhase::Failed {
                owner: MtpPhysicalOwner::Arena,
            },
        )
    }

    fn mark_failed(&mut self) {
        let owner = self.take_phase().into_owner();
        self.phase = MtpPhase::Failed { owner };
    }
}

enum MtpDrive {
    Waiting(DecoderWait),
    Complete(NativeProposal),
}

/// Generic MTP executor implementing the decoder proposal contract.
#[derive(Debug)]
pub struct MtpProposalExecutor<P> {
    source: Option<NativeProposalSource>,
    protocol: Option<MtpProtocol>,
    physical: Option<P>,
}

impl<P> MtpProposalExecutor<P>
where
    P: MtpPhysicalOps,
{
    pub fn new(prepared_plan_id: u64, protocol: MtpProtocol, physical: P) -> Result<Self> {
        if physical.stage_count() == 0 {
            return Err(mtp_error("executor requires at least one stage"));
        }
        if physical.protocol()? != protocol {
            return Err(mtp_error("protocol does not match the physical attachment"));
        }
        let source = NativeProposalSource {
            implementation: MTP_PROPOSAL_IMPLEMENTATION,
            prepared_plan_id,
            native_width: protocol.gamma,
        };
        source.validate()?;
        Ok(Self {
            source: Some(source),
            protocol: Some(protocol),
            physical: Some(physical),
        })
    }

    pub const fn target_only() -> Self {
        Self {
            source: None,
            protocol: None,
            physical: None,
        }
    }

    pub const fn protocol(&self) -> Option<&MtpProtocol> {
        self.protocol.as_ref()
    }

    fn enabled_protocol(&self) -> Result<&MtpProtocol> {
        self.protocol()
            .ok_or_else(|| mtp_error("target-only executor has no attachment"))
    }

    fn physical_mut(&mut self) -> Result<&mut P> {
        self.physical
            .as_mut()
            .ok_or_else(|| mtp_error("target-only executor has no attachment"))
    }

    fn progress(
        &mut self,
        context: &mut DecoderTransactionContext,
        state: &mut P::State,
        continuation: &mut MtpProposalContinuation<P>,
        kv: &mut P::KvView,
        resumed: bool,
    ) -> Result<MtpDrive> {
        self.physical_mut()?
            .validate_state(context.transaction(), state)?;
        let token_ids = self
            .enabled_protocol()?
            .draft_input_ids(continuation.anchor_token_id);
        loop {
            match continuation.take_phase() {
                MtpPhase::Initializing => {
                    if let Err(error) = self.physical_mut()?.initialize(
                        context.transaction(),
                        state,
                        kv,
                        continuation.anchor_token_id,
                        &token_ids,
                        &mut continuation.arena,
                    ) {
                        continuation.phase = MtpPhase::Initializing;
                        return Err(error);
                    }
                    continuation.phase = MtpPhase::StageReady { stage: 0 };
                }
                MtpPhase::StageReady { stage } => {
                    if stage < self.physical_mut()?.stage_count() {
                        match self.physical_mut()?.start_stage(
                            context.transaction(),
                            state,
                            kv,
                            stage,
                            continuation.sequence_tokens,
                            &token_ids,
                            &mut continuation.arena,
                        ) {
                            Ok(MtpStageStart::Waiting(current)) => {
                                continuation.phase = MtpPhase::StageWaiting {
                                    stage,
                                    continuation: current,
                                };
                                let MtpPhase::StageWaiting {
                                    continuation: current,
                                    ..
                                } = &continuation.phase
                                else {
                                    unreachable!()
                                };
                                let wait =
                                    self.physical_mut()?.stage_wait(context, stage, current)?;
                                return Ok(MtpDrive::Waiting(wait));
                            }
                            Ok(MtpStageStart::Complete) => {
                                continuation.phase = MtpPhase::StageReady { stage: stage + 1 };
                            }
                            Err(error) => {
                                continuation.phase = MtpPhase::StageReady { stage };
                                return Err(error);
                            }
                        }
                    } else {
                        match self.physical_mut()?.submit_head(
                            context.transaction(),
                            state,
                            kv,
                            continuation.anchor_token_id,
                            &mut continuation.arena,
                        ) {
                            Ok(mut download) => {
                                self.physical_mut()?
                                    .arm_head_completion(&mut download, context.completion_hub());
                                continuation.phase = MtpPhase::HeadWaiting { download };
                                return Ok(MtpDrive::Waiting(context.resolve_stage_wait(
                                    Vec::new(),
                                    crate::execution::WorkspaceClaim::NONE,
                                )?));
                            }
                            Err(error) => {
                                continuation.phase = MtpPhase::StageReady { stage };
                                return Err(error);
                            }
                        }
                    }
                }
                MtpPhase::StageWaiting {
                    stage,
                    continuation: mut current,
                } => {
                    if !resumed {
                        continuation.phase = MtpPhase::StageWaiting {
                            stage,
                            continuation: current,
                        };
                        return Err(Error::Internal {
                            message: "MTP stage was resumed without a wake".into(),
                        });
                    }
                    let lease = self
                        .physical_mut()?
                        .stage_needs_lease(&current)
                        .then(|| context.resume_lease())
                        .transpose()?;
                    match self.physical_mut()?.resume_stage(
                        context.transaction(),
                        state,
                        kv,
                        stage,
                        &mut current,
                        lease,
                        &mut continuation.arena,
                    ) {
                        Ok(MtpStageResume::Waiting) => {
                            continuation.phase = MtpPhase::StageWaiting {
                                stage,
                                continuation: current,
                            };
                            let MtpPhase::StageWaiting {
                                continuation: current,
                                ..
                            } = &continuation.phase
                            else {
                                unreachable!()
                            };
                            let wait = self.physical_mut()?.stage_wait(context, stage, current)?;
                            return Ok(MtpDrive::Waiting(wait));
                        }
                        Ok(MtpStageResume::Complete) => {
                            continuation.phase = MtpPhase::StageReady { stage: stage + 1 };
                        }
                        Err(error) => {
                            continuation.phase = MtpPhase::StageWaiting {
                                stage,
                                continuation: current,
                            };
                            return Err(error);
                        }
                    }
                }
                MtpPhase::HeadWaiting { mut download } => match self.physical_mut()?.poll_head(
                    context.transaction(),
                    kv,
                    &mut download,
                    &mut continuation.arena,
                ) {
                    Ok(Some(proposal)) => {
                        continuation.phase = MtpPhase::Finalizing;
                        return Ok(MtpDrive::Complete(proposal));
                    }
                    Ok(None) => {
                        self.physical_mut()?
                            .arm_head_completion(&mut download, context.completion_hub());
                        continuation.phase = MtpPhase::HeadWaiting { download };
                        return Ok(MtpDrive::Waiting(context.resolve_stage_wait(
                            Vec::new(),
                            crate::execution::WorkspaceClaim::NONE,
                        )?));
                    }
                    Err(error) => {
                        continuation.phase = MtpPhase::HeadWaiting { download };
                        return Err(error);
                    }
                },
                phase @ (MtpPhase::Failed { .. }
                | MtpPhase::CancelPending { .. }
                | MtpPhase::Cancelling { .. }) => {
                    continuation.phase = phase;
                    return Err(Error::Execution {
                        message: "failed or cancelling MTP must be cancelled".into(),
                    });
                }
                MtpPhase::Finalizing => {
                    continuation.phase = MtpPhase::Finalizing;
                    return Err(Error::Internal {
                        message: "completed MTP was resumed".into(),
                    });
                }
            }
        }
    }

    fn complete(
        &self,
        continuation: &MtpProposalContinuation<P>,
        proposal: NativeProposal,
    ) -> Result<NativeProposal> {
        proposal.validate_for_source(
            self.source
                .ok_or_else(|| mtp_error("target-only executor has no source"))?,
        )?;
        if !matches!(continuation.phase, MtpPhase::Finalizing) {
            return Err(Error::Internal {
                message: "MTP completed outside its finalizing phase".into(),
            });
        }
        Ok(proposal)
    }
}

impl<P> DecoderProposalExecutor<P::State, P::Backend> for MtpProposalExecutor<P>
where
    P: MtpPhysicalOps,
{
    type Continuation = MtpProposalContinuation<P>;

    fn source(&self) -> Result<Option<NativeProposalSource>> {
        Ok(self.source)
    }

    fn start(
        &mut self,
        context: &mut DecoderTransactionContext,
        state: &mut P::State,
        kv: &mut P::KvView,
        anchor_token_id: u32,
    ) -> Result<DecoderProposalProgress<Self::Continuation>> {
        if state.core().position() == 0 {
            return Ok(DecoderProposalProgress::FailedQuiescent(mtp_error(
                "proposal requires committed target context",
            )));
        }
        if let Err(error) = self
            .physical_mut()?
            .validate_state(context.transaction(), state)
        {
            return Ok(DecoderProposalProgress::FailedQuiescent(error));
        }
        let arena = match self.physical_mut()?.acquire_arena() {
            Ok(arena) => arena,
            Err(error) => return Ok(DecoderProposalProgress::FailedQuiescent(error)),
        };
        let mut continuation = MtpProposalContinuation {
            anchor_token_id,
            sequence_tokens: state.core().position(),
            arena,
            phase: MtpPhase::Initializing,
        };
        match self.progress(context, state, &mut continuation, kv, false) {
            Ok(MtpDrive::Complete(proposal)) => match self.complete(&continuation, proposal) {
                Ok(proposal) => Ok(DecoderProposalProgress::Complete(proposal)),
                Err(error) => {
                    continuation.mark_failed();
                    Ok(DecoderProposalProgress::FailedActive {
                        continuation,
                        error,
                    })
                }
            },
            Ok(MtpDrive::Waiting(wait)) => {
                Ok(DecoderProposalProgress::Waiting { continuation, wait })
            }
            Err(error) => {
                continuation.mark_failed();
                Ok(DecoderProposalProgress::FailedActive {
                    continuation,
                    error,
                })
            }
        }
    }

    fn resume(
        &mut self,
        context: &mut DecoderTransactionContext,
        state: &mut P::State,
        kv: &mut P::KvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderProposalResumeProgress> {
        match self.progress(context, state, continuation, kv, true) {
            Ok(MtpDrive::Complete(proposal)) => match self.complete(continuation, proposal) {
                Ok(proposal) => Ok(DecoderProposalResumeProgress::Complete(proposal)),
                Err(error) => {
                    continuation.mark_failed();
                    Ok(DecoderProposalResumeProgress::FailedActive(error))
                }
            },
            Ok(MtpDrive::Waiting(wait)) => Ok(DecoderProposalResumeProgress::Waiting(wait)),
            Err(error) => {
                continuation.mark_failed();
                Ok(DecoderProposalResumeProgress::FailedActive(error))
            }
        }
    }

    fn cancel(
        &mut self,
        context: &mut DecoderTransactionContext,
        state: &mut P::State,
        kv: &mut P::KvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderCancelProgress> {
        self.physical_mut()?
            .validate_state(context.transaction(), state)?;
        let phase = match continuation.take_phase() {
            MtpPhase::Cancelling { owner, quiescence } => {
                MtpPhase::Cancelling { owner, quiescence }
            }
            MtpPhase::CancelPending { owner } | MtpPhase::Failed { owner } => {
                MtpPhase::CancelPending { owner }
            }
            active => MtpPhase::CancelPending {
                owner: active.into_owner(),
            },
        };

        let (owner, mut quiescence) = match phase {
            MtpPhase::CancelPending { mut owner } => {
                match &mut owner {
                    MtpPhysicalOwner::Stage {
                        stage,
                        continuation: current,
                    } => match self
                        .physical_mut()?
                        .poll_stage_cancel_ready(*stage, current)
                    {
                        Ok(true) => {}
                        Ok(false) => {
                            continuation.phase = MtpPhase::CancelPending { owner };
                            return Ok(DecoderCancelProgress::Waiting);
                        }
                        Err(error) => {
                            continuation.phase = MtpPhase::CancelPending { owner };
                            return Err(error);
                        }
                    },
                    MtpPhysicalOwner::Head { download } => match self.physical_mut()?.poll_head(
                        context.transaction(),
                        kv,
                        download,
                        &mut continuation.arena,
                    ) {
                        Ok(Some(_)) => owner = MtpPhysicalOwner::Arena,
                        Ok(None) => {
                            self.physical_mut()?
                                .arm_head_completion(download, context.completion_hub());
                            continuation.phase = MtpPhase::CancelPending { owner };
                            return Ok(DecoderCancelProgress::Waiting);
                        }
                        Err(error) => {
                            continuation.phase = MtpPhase::CancelPending { owner };
                            return Err(error);
                        }
                    },
                    MtpPhysicalOwner::Arena => {}
                }
                let quiescence = match self.physical_mut()?.begin_quiescence() {
                    Ok(quiescence) => quiescence,
                    Err(error) => {
                        continuation.phase = MtpPhase::CancelPending { owner };
                        return Err(error);
                    }
                };
                continuation.phase = MtpPhase::Cancelling { owner, quiescence };
                return Ok(DecoderCancelProgress::Waiting);
            }
            MtpPhase::Cancelling { owner, quiescence } => (owner, quiescence),
            _ => unreachable!(),
        };

        match self.physical_mut()?.poll_quiescence(&mut quiescence) {
            Ok(true) => {}
            Ok(false) => {
                continuation.phase = MtpPhase::Cancelling { owner, quiescence };
                return Ok(DecoderCancelProgress::Waiting);
            }
            Err(error) => {
                continuation.phase = MtpPhase::Cancelling { owner, quiescence };
                return Err(error);
            }
        }
        if let MtpPhysicalOwner::Stage {
            stage,
            continuation: mut current,
        } = owner
            && let Err(error) =
                self.physical_mut()?
                    .cancel_stage(stage, &mut current, &mut continuation.arena)
        {
            continuation.phase = MtpPhase::CancelPending {
                owner: MtpPhysicalOwner::Stage {
                    stage,
                    continuation: current,
                },
            };
            return Err(error);
        }
        Ok(DecoderCancelProgress::Complete)
    }

    fn retain_provisional(
        &mut self,
        context: &mut DecoderProvisionalRetainContext<'_, P::State, P::Backend>,
    ) -> Result<()> {
        self.physical_mut()?.retain_provisional(context)
    }
}

fn mtp_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("MTP: {}", message.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_freezes_anchor_noise_and_bonus_rows() {
        let protocol = MtpConfig {
            block_size: 4,
            noise_token_id: Some(7),
            target_layer_ids: vec![1, 3],
            markov_rank: Some(2),
        }
        .protocol()
        .unwrap();
        assert_eq!(protocol.draft_input_ids(11), [11, 7, 7, 7]);
        assert_eq!(protocol.target_rows_for_drafts(4).unwrap(), 5);
        assert_eq!(protocol.max_external_commit_tokens, 5);
    }
}
