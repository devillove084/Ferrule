//! Bounded host DTOs. Deserialization is not validation: every frame is rebuilt
//! through the domain constructors before dispatch. No physical KV authority,
//! sequence topology, arena identity or device address belongs in this protocol.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ferrule_common::execution::*;
use ferrule_common::{
    ParallelRankId, ParallelTopologyId, ParticipantSet, ValidatedParallelTopology,
};
use ferrule_model::decoder::{
    DecoderKvPageSnapshot, DecoderKvPageStatus, DenseLogits, KvCommitBinding, KvEndProgress,
};
use ferrule_model::transformer::{HostRows, RowsDType, RowsShape, SegmentInput, SegmentOutput};
use serde::{Deserialize, Serialize};

use super::decoder::{DecoderBoot, Result, error};
use crate::SessionId;
use crate::parallel::pipeline::{
    PipelineCommand, PipelineCommandKey, PipelinePreparedProjection, PipelineRank, PipelineReply,
    PipelineStageDescription,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyFrame {
    pub transaction: u64,
    pub topology: u32,
    pub world_size: u32,
    pub plan: ferrule_common::ParallelismPlan,
    pub participants: Vec<u32>,
    pub generation: u64,
    pub rank: u32,
    pub session: u64,
}
impl KeyFrame {
    pub fn encode(key: &PipelineCommandKey) -> Self {
        let binding = key.binding();
        Self {
            transaction: binding.transaction().get(),
            topology: binding.topology_id().get(),
            world_size: binding.participants().world_size(),
            plan: binding.participants().plan(),
            participants: binding
                .participants()
                .iter()
                .map(ParallelRankId::get)
                .collect(),
            generation: binding.generation(),
            rank: key.rank().global.get(),
            session: key.session().0,
        }
    }
    pub fn decode(&self) -> Result<PipelineCommandKey> {
        let rank = ParallelRankId::new(self.rank);
        let topology = ValidatedParallelTopology::new(
            ParallelTopologyId::new(self.topology),
            self.world_size,
            rank,
            self.plan,
        )
        .map_err(|e| error(format!("key topology: {e:?}")))?;
        let participants = ParticipantSet::new(
            &topology,
            self.participants.iter().copied().map(ParallelRankId::new),
        )
        .map_err(|e| error(format!("key participants: {e:?}")))?;
        PipelineCommandKey::new(
            KvCommitBinding::new(
                ExecutionTransactionId::new(self.transaction)?,
                topology.topology_id(),
                participants,
                self.generation,
            )?,
            PipelineRank {
                local: rank,
                global: rank,
            },
            SessionId(self.session),
        )
    }
}

/// The serial pipeline accepts exactly one committed prefill/decode sequence.
/// Keeping this subset explicit avoids silently dropping intent or logits flags.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchFrame {
    pub decode: bool,
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub writes: Vec<Option<u32>>,
    pub full_logits: Vec<bool>,
    pub state_slot: u32,
    pub context_len: u32,
    pub sequence_len: u32,
    pub blocks: Vec<u32>,
}
impl BatchFrame {
    pub fn encode(batch: &ExecutionBatch) -> Result<Self> {
        let [sequence] = batch.sequences() else {
            return Err(error("wire requires one sequence"));
        };
        let decode = batch.mode() == ForwardMode::Decode;
        if batch.intent() != ExecutionIntent::Committed
            || !matches!(batch.mode(), ForwardMode::Prefill | ForwardMode::Decode)
            || sequence.phase
                != if decode {
                    ForwardPhase::Decode
                } else {
                    ForwardPhase::Prefill
                }
            || sequence.query != (0..u32::try_from(batch.len()).map_err(error)?)
            || sequence.block_table
                != (0..u32::try_from(batch.kv_block_ids().len()).map_err(error)?)
        {
            return Err(error("unsupported serial batch projection"));
        }
        Ok(Self {
            decode,
            tokens: batch.token_ids().to_vec(),
            positions: batch.positions().to_vec(),
            writes: batch
                .kv_write_slots()
                .iter()
                .map(|s| s.map(KvWriteSlot::get))
                .collect(),
            full_logits: batch
                .logits()
                .iter()
                .map(|r| match r {
                    LogitsRequest::Full => Ok(true),
                    LogitsRequest::None => Ok(false),
                    _ => Err(error("wire does not support top-k requests")),
                })
                .collect::<Result<_>>()?,
            state_slot: sequence.state_slot.get(),
            context_len: sequence.context_len,
            sequence_len: sequence.sequence_len,
            blocks: batch.kv_block_ids().iter().map(|b| b.get()).collect(),
        })
    }
    pub fn decode(&self, description: &PipelineStageDescription) -> Result<ExecutionBatch> {
        if self.tokens.len() > description.config.max_batch_tokens
            || self.blocks.len() > description.config.max_pages
            || self
                .tokens
                .iter()
                .any(|&t| t as usize >= description.vocabulary)
        {
            return Err(error("batch frame exceeds decoder bounds"));
        }
        let phase = if self.decode {
            ForwardPhase::Decode
        } else {
            ForwardPhase::Prefill
        };
        let batch = ExecutionBatch::new(
            if self.decode {
                ForwardMode::Decode
            } else {
                ForwardMode::Prefill
            },
            self.tokens.clone(),
            self.positions.clone(),
            self.writes
                .iter()
                .map(|s| s.map(KvWriteSlot::new))
                .collect(),
            self.full_logits
                .iter()
                .map(|&full| {
                    if full {
                        LogitsRequest::Full
                    } else {
                        LogitsRequest::None
                    }
                })
                .collect(),
            vec![ExecutionSequence::new(
                StateSlot::new(self.state_slot),
                phase,
                0..u32::try_from(self.tokens.len()).map_err(error)?,
                self.context_len,
                self.sequence_len,
                0..u32::try_from(self.blocks.len()).map_err(error)?,
            )],
            self.blocks.iter().copied().map(KvBlockId::new).collect(),
        );
        batch.validate(1, &description.execution_capabilities()?)?;
        Ok(batch)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReservationFrame {
    pub state_slot: u32,
    pub execution_state_slot: u32,
    pub positions: Range<usize>,
    pub new_pages: Vec<u32>,
    pub generation: u64,
    pub execution_generation: u64,
    /// Logical tail index, source page, replacement page (never physical slots).
    pub cow: Option<(usize, u32, u32)>,
}
impl ReservationFrame {
    pub fn encode(r: &KvReservationView) -> Self {
        Self {
            state_slot: r.state_slot.get(),
            execution_state_slot: r.execution_state_slot.get(),
            positions: r.positions.clone(),
            new_pages: r.newly_allocated.iter().map(|p| p.0).collect(),
            generation: r.generation,
            execution_generation: r.execution_generation,
            cow: r
                .cow_replacement
                .map(|c| (c.logical_page, c.source.0, c.replacement.0)),
        }
    }
    pub fn decode(&self, description: &PipelineStageDescription) -> Result<KvReservationView> {
        if self.positions.is_empty()
            || self.positions.end > description.config.max_positions
            || self.new_pages.len() > description.config.max_pages
        {
            return Err(error("invalid wire reservation bounds"));
        }
        Ok(KvReservationView {
            state_slot: StateSlot::new(self.state_slot),
            execution_state_slot: StateSlot::new(self.execution_state_slot),
            positions: self.positions.clone(),
            newly_allocated: self.new_pages.iter().copied().map(KvPageId).collect(),
            generation: self.generation,
            execution_generation: self.execution_generation,
            cow_replacement: self
                .cow
                .map(|(logical_page, source, replacement)| KvCowReplacement {
                    logical_page,
                    source: KvPageId(source),
                    replacement: KvPageId(replacement),
                }),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowsFrame {
    pub rows: usize,
    pub width: usize,
    pub bf16: bool,
    pub values: Vec<f32>,
}
impl RowsFrame {
    fn encode(rows: &HostRows) -> Result<Self> {
        finite(rows.values())?;
        Ok(Self {
            rows: rows.shape().rows(),
            width: rows.shape().width(),
            bf16: rows.dtype() == RowsDType::Bf16,
            values: rows.values().to_vec(),
        })
    }
    fn decode(self, description: &PipelineStageDescription) -> Result<HostRows> {
        if self.rows == 0
            || self.rows > description.config.max_batch_tokens
            || self.width != description.hidden
        {
            return Err(error("hidden frame geometry mismatch"));
        }
        finite(&self.values)?;
        HostRows::new(
            RowsShape::new(self.rows, self.width)?,
            if self.bf16 {
                RowsDType::Bf16
            } else {
                RowsDType::F32
            },
            None,
            self.values,
        )
    }
}
pub(super) fn finite(values: &[f32]) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        Err(error("nonfinite wire values"))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum InputFrame {
    Tokens,
    Hidden { next_layer: usize, rows: RowsFrame },
}
impl InputFrame {
    fn encode(input: &SegmentInput) -> Result<Self> {
        Ok(match input {
            SegmentInput::Tokens => Self::Tokens,
            SegmentInput::Hidden { next_layer, rows } => Self::Hidden {
                next_layer: *next_layer,
                rows: RowsFrame::encode(rows)?,
            },
        })
    }
    fn decode(self, d: &PipelineStageDescription) -> Result<SegmentInput> {
        match self {
            Self::Tokens => Ok(SegmentInput::Tokens),
            Self::Hidden { next_layer, rows } => Ok(SegmentInput::Hidden {
                next_layer,
                rows: rows.decode(d)?,
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DecoderCommand {
    Describe,
    Stats,
    Create {
        session: u64,
    },
    Fork {
        source: u64,
        target: u64,
    },
    Release {
        session: u64,
        pages: Vec<u32>,
    },
    Prepare {
        key: KeyFrame,
        batch: BatchFrame,
        reservation: ReservationFrame,
    },
    Execute {
        key: KeyFrame,
        input: InputFrame,
        cancelled: bool,
    },
    Ready {
        key: KeyFrame,
    },
    Install {
        key: KeyFrame,
        generation: u64,
    },
    Poll {
        key: KeyFrame,
        generation: u64,
    },
    Abort {
        key: KeyFrame,
        generation: u64,
    },
    Rollback {
        key: KeyFrame,
    },
    Publish {
        key: KeyFrame,
    },
    CheckRetirement {
        key: KeyFrame,
        pages: Vec<u32>,
    },
    Retire {
        key: KeyFrame,
        generation: u64,
        pages: Vec<u32>,
    },
    Finish {
        key: KeyFrame,
    },
}
impl DecoderCommand {
    pub fn key(&self) -> Option<&KeyFrame> {
        match self {
            Self::Prepare { key, .. }
            | Self::Execute { key, .. }
            | Self::Ready { key }
            | Self::Install { key, .. }
            | Self::Poll { key, .. }
            | Self::Abort { key, .. }
            | Self::Rollback { key }
            | Self::Publish { key }
            | Self::CheckRetirement { key, .. }
            | Self::Retire { key, .. }
            | Self::Finish { key } => Some(key),
            _ => None,
        }
    }
    pub fn session(&self) -> u64 {
        match self {
            Self::Create { session } | Self::Release { session, .. } => *session,
            Self::Fork { target, .. } => *target,
            _ => self.key().map_or(0, |k| k.session),
        }
    }
    pub fn encode(command: &PipelineCommand) -> Result<Self> {
        use PipelineCommand as C;
        Ok(match command {
            C::Describe => Self::Describe,
            C::Stats => Self::Stats,
            C::Create { session } => Self::Create { session: session.0 },
            C::Fork { source, target } => Self::Fork {
                source: source.0,
                target: target.0,
            },
            C::Release { session, pages } => Self::Release {
                session: session.0,
                pages: pages.iter().map(|p| p.0).collect(),
            },
            C::Prepare {
                key,
                batch,
                reservation,
            } => Self::Prepare {
                key: KeyFrame::encode(key),
                batch: BatchFrame::encode(batch)?,
                reservation: ReservationFrame::encode(reservation),
            },
            C::Execute {
                key,
                input,
                cancellation,
            } => Self::Execute {
                key: KeyFrame::encode(key),
                input: InputFrame::encode(input)?,
                cancelled: cancellation.load(Ordering::Acquire),
            },
            C::Ready(key) => Self::Ready {
                key: KeyFrame::encode(key),
            },
            C::Install { key, generation } => Self::Install {
                key: KeyFrame::encode(key),
                generation: *generation,
            },
            C::PollInstall { key, generation } => Self::Poll {
                key: KeyFrame::encode(key),
                generation: *generation,
            },
            C::Abort { key, generation } => Self::Abort {
                key: KeyFrame::encode(key),
                generation: *generation,
            },
            C::Rollback(key) => Self::Rollback {
                key: KeyFrame::encode(key),
            },
            C::Publish(key) => Self::Publish {
                key: KeyFrame::encode(key),
            },
            C::CheckRetirement { key, pages } => Self::CheckRetirement {
                key: KeyFrame::encode(key),
                pages: pages.iter().map(|p| p.0).collect(),
            },
            C::Retire {
                key,
                generation,
                pages,
            } => Self::Retire {
                key: KeyFrame::encode(key),
                generation: *generation,
                pages: pages.iter().map(|p| p.0).collect(),
            },
            C::Finish(key) => Self::Finish {
                key: KeyFrame::encode(key),
            },
        })
    }
    pub fn decode(self, d: &PipelineStageDescription) -> Result<PipelineCommand> {
        use PipelineCommand as C;
        let pages = |p: Vec<u32>| -> Result<Vec<KvPageId>> {
            if p.len() > d.config.max_pages
                || p.iter().collect::<std::collections::BTreeSet<_>>().len() != p.len()
            {
                return Err(error("invalid wire page list"));
            }
            Ok(p.into_iter().map(KvPageId).collect())
        };
        Ok(match self {
            Self::Describe => C::Describe,
            Self::Stats => C::Stats,
            Self::Create { session } => C::Create {
                session: SessionId(session),
            },
            Self::Fork { source, target } => C::Fork {
                source: SessionId(source),
                target: SessionId(target),
            },
            Self::Release { session, pages: p } => C::Release {
                session: SessionId(session),
                pages: pages(p)?,
            },
            Self::Prepare {
                key,
                batch,
                reservation,
            } => C::Prepare {
                key: key.decode()?,
                batch: Box::new(batch.decode(d)?),
                reservation: reservation.decode(d)?,
            },
            Self::Execute {
                key,
                input,
                cancelled,
            } => C::Execute {
                key: key.decode()?,
                input: input.decode(d)?,
                cancellation: Arc::new(AtomicBool::new(cancelled)),
            },
            Self::Ready { key } => C::Ready(key.decode()?),
            Self::Install { key, generation } => C::Install {
                key: key.decode()?,
                generation,
            },
            Self::Poll { key, generation } => C::PollInstall {
                key: key.decode()?,
                generation,
            },
            Self::Abort { key, generation } => C::Abort {
                key: key.decode()?,
                generation,
            },
            Self::Rollback { key } => C::Rollback(key.decode()?),
            Self::Publish { key } => C::Publish(key.decode()?),
            Self::CheckRetirement { key, pages: p } => C::CheckRetirement {
                key: key.decode()?,
                pages: pages(p)?,
            },
            Self::Retire {
                key,
                generation,
                pages: p,
            } => C::Retire {
                key: key.decode()?,
                generation,
                pages: pages(p)?,
            },
            Self::Finish { key } => C::Finish(key.decode()?),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgressFrame {
    Pending,
    Complete,
    ConsumedRejected,
}
impl From<KvEndProgress> for ProgressFrame {
    fn from(p: KvEndProgress) -> Self {
        match p {
            KvEndProgress::Pending => Self::Pending,
            KvEndProgress::Complete => Self::Complete,
            KvEndProgress::ConsumedRejected => Self::ConsumedRejected,
        }
    }
}
impl From<ProgressFrame> for KvEndProgress {
    fn from(p: ProgressFrame) -> Self {
        match p {
            ProgressFrame::Pending => Self::Pending,
            ProgressFrame::Complete => Self::Complete,
            ProgressFrame::ConsumedRejected => Self::ConsumedRejected,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageStatusFrame {
    Vacant,
    Resident,
    Preempted,
}

/// Process statistics deliberately report PID, not a fabricated Rust ThreadId.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderProcessStats {
    pub pid: u32,
    pub rank: u32,

    pub sessions: usize,
    pub executions: usize,
    pub physical_pages: usize,
    pub resident_pages: usize,
    pub free_pages: usize,
    pub active_transactions: usize,
    pub expert_outstanding: usize,
    pub experts: Vec<super::decoder_expert::ExpertProcessStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DecoderReply {
    Description {
        boot: DecoderBoot,
        hidden: usize,
        vocabulary: usize,
        kv_heads: usize,
        head_dim: usize,
    },
    Stats(DecoderProcessStats),
    Generation {
        generation: u64,
    },
    Prepared {
        batch: BatchFrame,
        reservation: ReservationFrame,
        page_size: usize,
        statuses: Vec<(u32, PageStatusFrame)>,
    },
    Hidden {
        next_layer: usize,
        rows: RowsFrame,
    },
    Logits {
        rows: usize,
        width: usize,
        values: Vec<f32>,
    },
    Ack {
        progress: ProgressFrame,
    },
}
impl DecoderReply {
    pub(super) fn encode(reply: PipelineReply, boot: &DecoderBoot) -> Result<Self> {
        Ok(match reply {
            PipelineReply::Description(d) => Self::Description {
                boot: boot.clone(),
                hidden: d.hidden,
                vocabulary: d.vocabulary,
                kv_heads: d.kv_heads,
                head_dim: d.head_dim,
            },
            PipelineReply::Stats(s) => Self::Stats(DecoderProcessStats {
                pid: std::process::id(),
                rank: s.rank.global.get(),

                sessions: s.sessions,
                executions: s.executions,
                physical_pages: s.kv.physical_pages,
                resident_pages: s.kv.resident_pages,
                free_pages: s.kv.free_pages,
                active_transactions: s.kv.active_transactions,
                expert_outstanding: s.expert_outstanding,
                experts: Vec::new(),
            }),
            PipelineReply::Generation(generation) => Self::Generation { generation },
            PipelineReply::Prepared { projection: p } => Self::Prepared {
                batch: BatchFrame::encode(&p.batch)?,
                reservation: ReservationFrame::encode(&p.reservation),
                page_size: p.page_size,
                statuses: p
                    .page_statuses
                    .into_iter()
                    .map(|s| {
                        (
                            s.page.0,
                            match s.status {
                                DecoderKvPageStatus::Vacant => PageStatusFrame::Vacant,
                                DecoderKvPageStatus::Resident => PageStatusFrame::Resident,
                                DecoderKvPageStatus::Preempted => PageStatusFrame::Preempted,
                            },
                        )
                    })
                    .collect(),
            },
            PipelineReply::Executed {
                output: SegmentOutput::Hidden { next_layer, rows },
            } => Self::Hidden {
                next_layer,
                rows: RowsFrame::encode(&rows)?,
            },
            PipelineReply::Executed {
                output: SegmentOutput::Logits(l),
            } => {
                finite(l.values())?;
                Self::Logits {
                    rows: l.rows(),
                    width: l.width(),
                    values: l.values().to_vec(),
                }
            }
            PipelineReply::Ack(p) => Self::Ack { progress: p.into() },
        })
    }
    pub(super) fn decode(
        self,
        command: &DecoderCommand,
        d: &PipelineStageDescription,
        boot: &DecoderBoot,
    ) -> Result<PipelineReply> {
        let reply = match (self, command) {
            (
                Self::Description {
                    boot: actual,
                    hidden,
                    vocabulary,
                    kv_heads,
                    head_dim,
                },
                DecoderCommand::Describe,
            ) if actual == *boot => {
                let description = boot.description(hidden, vocabulary, kv_heads, head_dim)?;
                if description != *d {
                    return Err(error("decoder description changed"));
                }
                PipelineReply::Description(description)
            }
            (
                Self::Generation { generation },
                DecoderCommand::Create { .. } | DecoderCommand::Fork { .. },
            ) => PipelineReply::Generation(generation),
            (
                Self::Prepared {
                    batch,
                    reservation,
                    page_size,
                    statuses,
                },
                DecoderCommand::Prepare {
                    key,
                    batch: expected,
                    reservation: wanted,
                },
            ) => {
                let owner = boot.stage_boot()?;
                let key = key.decode()?;
                if key.rank() != owner.rank
                    || boot.description(d.hidden, d.vocabulary, d.kv_heads, d.head_dim)? != *d
                {
                    return Err(error("prepared reply owner/capacity differs from boot"));
                }
                if &batch != expected
                    || &reservation != wanted
                    || statuses.len() > d.config.max_pages
                {
                    return Err(error("prepared frame differs from request"));
                }
                let projection = PipelinePreparedProjection {
                    batch: batch.decode(d)?,
                    reservation: reservation.decode(d)?,
                    page_size,
                    page_statuses: statuses
                        .into_iter()
                        .map(|(p, s)| DecoderKvPageSnapshot {
                            page: KvPageId(p),
                            status: match s {
                                PageStatusFrame::Vacant => DecoderKvPageStatus::Vacant,
                                PageStatusFrame::Resident => DecoderKvPageStatus::Resident,
                                PageStatusFrame::Preempted => DecoderKvPageStatus::Preempted,
                            },
                        })
                        .collect(),
                };
                projection.clone().into_commit_projection(
                    &expected.decode(d)?,
                    &wanted.decode(d)?,
                    d,
                )?;
                PipelineReply::Prepared { projection }
            }
            (Self::Hidden { next_layer, rows }, DecoderCommand::Execute { .. }) => {
                PipelineReply::Executed {
                    output: SegmentOutput::Hidden {
                        next_layer,
                        rows: rows.decode(d)?,
                    },
                }
            }
            (
                Self::Logits {
                    rows,
                    width,
                    values,
                },
                DecoderCommand::Execute { .. },
            ) => {
                if rows == 0 || rows > d.config.max_batch_tokens || width != d.vocabulary {
                    return Err(error("invalid logits frame shape"));
                }
                finite(&values)?;
                PipelineReply::Executed {
                    output: SegmentOutput::Logits(DenseLogits::new(rows, width, values)?),
                }
            }
            (
                Self::Ack { progress },
                DecoderCommand::Release { .. }
                | DecoderCommand::Ready { .. }
                | DecoderCommand::Install { .. }
                | DecoderCommand::Poll { .. }
                | DecoderCommand::Abort { .. }
                | DecoderCommand::Rollback { .. }
                | DecoderCommand::Publish { .. }
                | DecoderCommand::CheckRetirement { .. }
                | DecoderCommand::Retire { .. }
                | DecoderCommand::Finish { .. },
            ) => PipelineReply::Ack(progress.into()),
            _ => {
                return Err(error(
                    "unexpected decoder reply (use process_stats for process identities)",
                ));
            }
        };
        if let PipelineReply::Executed { output } = &reply {
            let rows = match output {
                SegmentOutput::Hidden { rows, .. } => rows.shape().rows(),
                SegmentOutput::Logits(l) => l.rows(),
            };
            d.validate_output(output, rows)?;
        }
        Ok(reply)
    }
}

#[cfg(test)]
mod projection_tests {
    use super::super::decoder::{
        DECODER_WIRE_VERSION, DecoderDevice, DecoderPrecision, DecoderRecipeKind, KvConfigFrame,
        SegmentFrame,
    };
    use super::*;
    use crate::parallel::pipeline::PipelineConfig;
    use ferrule_common::ParallelismPlan;
    use ferrule_model::execution::ExecutionPrecisionPolicy;

    fn fixture() -> (
        DecoderBoot,
        PipelineStageDescription,
        DecoderCommand,
        DecoderReply,
    ) {
        let config = PipelineConfig {
            page_size: 4,
            max_pages: 4,
            max_positions: 16,
            max_batch_tokens: 4,
            session_capacity: 1,
            max_parameter_bytes: 1024,
            max_ack_polls: 2,
            precision: ExecutionPrecisionPolicy::f32(),
        };
        let boot = DecoderBoot {
            version: DECODER_WIRE_VERSION,
            rank: 0,
            checkpoint: "metadata-only-fixture".into(),
            recipe: DecoderRecipeKind::Synthetic,
            segment: SegmentFrame {
                total_layers: 1,
                layers: 0..1,
                embedding: true,
                output: true,
            },
            precision: DecoderPrecision::F32,
            device: DecoderDevice::Cpu,
            kv: KvConfigFrame::encode(config),
            experts: None,
        };
        let description = boot.description(4, 8, 1, 4).unwrap();
        let batch = BatchFrame {
            decode: false,
            tokens: vec![1, 2],
            positions: vec![0, 1],
            writes: vec![Some(0), Some(1)],
            full_logits: vec![false, true],
            state_slot: 0,
            context_len: 0,
            sequence_len: 2,
            blocks: vec![0],
        };
        let reservation = ReservationFrame {
            state_slot: 0,
            execution_state_slot: 0,
            positions: 0..2,
            new_pages: vec![0],
            generation: 0,
            execution_generation: 0,
            cow: None,
        };
        let key = KeyFrame {
            transaction: 1,
            topology: 9,
            world_size: 1,
            plan: ParallelismPlan::validated(1, 1, 1, 1, 1, 1).unwrap(),
            participants: vec![0],
            generation: 1,
            rank: 0,
            session: 7,
        };
        let command = DecoderCommand::Prepare {
            key,
            batch: batch.clone(),
            reservation: reservation.clone(),
        };
        let reply = DecoderReply::Prepared {
            batch,
            reservation,
            page_size: 4,
            statuses: vec![(0, PageStatusFrame::Vacant)],
        };
        (boot, description, command, reply)
    }

    #[test]
    fn prepared_wire_roundtrip_keeps_full_request_and_capacity() {
        let (boot, description, command, reply) = fixture();
        let encoded = serde_json::to_vec(&reply).unwrap();
        let reply: DecoderReply = serde_json::from_slice(&encoded).unwrap();
        let PipelineReply::Prepared { projection } =
            reply.decode(&command, &description, &boot).unwrap()
        else {
            panic!("prepared")
        };
        let DecoderCommand::Prepare {
            key,
            batch,
            reservation,
        } = command
        else {
            unreachable!()
        };
        let batch = batch.decode(&description).unwrap();
        let reservation = reservation.decode(&description).unwrap();
        let weak = projection
            .clone()
            .into_commit_projection(&batch, &reservation, &description)
            .unwrap();
        let strong = projection
            .into_execution_identity(&batch, &reservation, &description, &[key.session])
            .unwrap();
        assert_eq!(strong.commit_projection(), weak);
        assert_eq!(strong.rows(), 2);
        assert_eq!(boot.kv.max_pages, 4);
        assert_eq!(boot.kv.physical_pages, 4);
        assert_eq!(DECODER_WIRE_VERSION, 3);
    }

    #[test]
    fn prepared_wire_rejects_every_batch_reservation_and_snapshot_dimension() {
        for axis in 0..22 {
            let (boot, d, command, mut reply) = fixture();
            let DecoderReply::Prepared {
                batch,
                reservation,
                page_size,
                statuses,
            } = &mut reply
            else {
                unreachable!()
            };
            match axis {
                0 => batch.tokens[0] += 1,
                1 => batch.positions[0] += 1,
                2 => batch.writes[0] = Some(4),
                3 => batch.full_logits[0] = true,
                4 => batch.decode = true,
                5 => batch.state_slot = 1,
                6 => batch.context_len = 1,
                7 => batch.sequence_len = 3,
                8 => batch.blocks[0] = 1,
                9 => reservation.state_slot = 1,
                10 => reservation.execution_state_slot = 1,
                11 => reservation.positions = 1..3,
                12 => reservation.new_pages[0] = 1,
                13 => reservation.generation += 1,
                14 => reservation.execution_generation += 1,
                15 => reservation.cow = Some((0, 0, 1)),
                16 => *page_size = 8,
                17 => statuses.clear(),
                18 => statuses.push((0, PageStatusFrame::Vacant)),
                19 => statuses[0].1 = PageStatusFrame::Resident,
                20 => statuses[0].1 = PageStatusFrame::Preempted,
                21 => statuses.push((1, PageStatusFrame::Resident)),
                _ => unreachable!(),
            }
            assert!(
                reply.decode(&command, &d, &boot).is_err(),
                "wire axis {axis}"
            );
        }
    }

    #[test]
    fn prepared_wire_rejects_old_version_capacity_and_wrong_owner() {
        for axis in 0..5 {
            let (mut boot, d, mut command, reply) = fixture();
            match axis {
                0 => boot.version = 1,
                1 => boot.kv.physical_pages += 1,
                2 => boot.kv.max_pages += 1,
                3 => boot.rank = 1,
                4 => {
                    if let DecoderCommand::Prepare { key, .. } = &mut command {
                        key.transaction = 0;
                    }
                }
                _ => unreachable!(),
            }
            assert!(
                reply.decode(&command, &d, &boot).is_err(),
                "boot axis {axis}"
            );
        }
        let (boot, _, _, _) = fixture();
        let mut json = serde_json::to_value(boot).unwrap();
        json["kv"].as_object_mut().unwrap().remove("physical_pages");
        assert!(serde_json::from_value::<DecoderBoot>(json).is_err());
    }

    #[test]
    fn serial_wire_rejects_unrepresentable_intent_phase_logits_and_shapes() {
        let (_, d, command, _) = fixture();
        let DecoderCommand::Prepare { batch, .. } = command else {
            unreachable!()
        };
        let source = batch.decode(&d).unwrap();
        assert!(
            BatchFrame::encode(
                &source
                    .clone()
                    .with_intent(ExecutionIntent::ProvisionalVerification)
            )
            .is_err()
        );
        for axis in 0..4 {
            let mut sequences = source.sequences().to_vec();
            let mut logits = source.logits().to_vec();
            match axis {
                0 => sequences[0].phase = ForwardPhase::Decode,
                1 => sequences[0].query = 1..2,
                2 => sequences[0].block_table = 1..2,
                3 => logits[0] = LogitsRequest::TopK(std::num::NonZeroU32::new(1).unwrap()),
                _ => unreachable!(),
            }
            let changed = ExecutionBatch::new(
                source.mode(),
                source.token_ids().to_vec(),
                source.positions().to_vec(),
                source.kv_write_slots().to_vec(),
                logits,
                sequences,
                source.kv_block_ids().to_vec(),
            );
            assert!(BatchFrame::encode(&changed).is_err());
        }
        for axis in 0..7 {
            let mut changed = batch.clone();
            match axis {
                0 => changed.positions.pop().map(|_| ()).unwrap(),
                1 => changed.writes.pop().map(|_| ()).unwrap(),
                2 => changed.full_logits.pop().map(|_| ()).unwrap(),
                3 => changed.state_slot = u32::MAX,
                4 => changed.context_len = u32::MAX,
                5 => changed.tokens[0] = d.vocabulary as u32,
                6 => changed.blocks.clear(),
                _ => unreachable!(),
            }
            assert!(changed.decode(&d).is_err(), "malformed batch axis {axis}");
        }
    }
}
