//! Independent PP -> EP pipes. No message is routed back through the root
//! while it waits for a stage. EP children inherit the PP isolation group, so
//! root timeout/quarantine terminates the entire tree, including stuck experts.

use std::collections::{BTreeMap, BTreeSet};

use std::sync::Arc;
use std::time::Duration;

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{ParallelRankId, Result};
use ferrule_model::TensorRole;
use ferrule_model::moe::ExpertId;
use ferrule_model::transformer::{
    BoundDecoderResources, CpuReferenceExpertWorker, ExpertAvailability, ExpertProvider,
    ExpertResult, ExpertToken, ExpertTokenBucket, ExpertWorker, FeedForward, PreparedLinear,
    PreparedSwiGlu, StateDictMaterializer,
};
use serde::{Deserialize, Serialize};

use super::decoder::{DecoderBoot, DecoderDevice, error};
use super::decoder_wire::finite;
use super::*;
use crate::parallel::data::PanicQuiescence;
use crate::parallel::expert::ExpertGroup;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertBoot {
    pub expert_boot: DecoderBoot,
    pub owner: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertTokenFrame {
    pub transaction: u64,
    pub sequence: u64,
    pub source: u32,
    pub row: usize,
    pub route: usize,
    pub layer: usize,
    pub expert: usize,
    pub weight: f32,
    pub values: Vec<f32>,
}
impl ExpertTokenFrame {
    fn encode(t: &ExpertToken) -> Self {
        Self {
            transaction: t.transaction.get(),
            sequence: t.sequence,
            source: t.source_rank.get(),
            row: t.source_row,
            route: t.route_slot,
            layer: t.expert.layer,
            expert: t.expert.expert,
            weight: t.weight,
            values: t.payload.clone(),
        }
    }
    fn decode(self) -> Result<ExpertToken> {
        finite(&self.values)?;
        if !self.weight.is_finite() {
            return Err(error("nonfinite route weight"));
        }
        Ok(ExpertToken {
            transaction: ExecutionTransactionId::new(self.transaction)?,
            sequence: self.sequence,
            source_rank: ParallelRankId::new(self.source),
            source_row: self.row,
            route_slot: self.route,
            expert: ExpertId::new(self.layer, self.expert),
            weight: self.weight,
            payload: self.values,
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExpertCommand {
    Stats,
    Compute {
        transaction: u64,
        source: u32,
        layer: usize,
        tokens: Vec<ExpertTokenFrame>,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertProcessStats {
    pub pid: u32,
    pub owner: u32,
    pub device: DecoderDevice,
    pub owned_experts: Vec<(usize, usize)>,
    pub calls: usize,
    pub tokens: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExpertReply {
    Stats(ExpertProcessStats),
    // Identities are echoed only after the real compute returns. Outputs are
    // unweighted; the existing model dispatch plan validates and combines them.
    Results {
        owner: u32,
        tokens: Vec<ExpertTokenFrame>,
    },
}

struct LocalExperts(BTreeMap<ExpertId, Arc<PreparedSwiGlu>>);
impl ExpertProvider for LocalExperts {
    fn expert(&mut self, layer: usize, expert: usize) -> Result<ExpertAvailability> {
        self.0
            .get(&ExpertId::new(layer, expert))
            .cloned()
            .map(ExpertAvailability::Ready)
            .ok_or_else(|| error("nonlocal expert requested"))
    }
}
fn prepare(
    resources: &BoundDecoderResources,
    group: &ExpertGroup,
    owner: ParallelRankId,
    max: u64,
) -> Result<LocalExperts> {
    group.validate_resources(resources)?;
    let materializer = StateDictMaterializer::new(max)?;
    let mut experts = BTreeMap::new();
    for layer in group.layers.clone() {
        let FeedForward::Moe(moe) = resources.spec().layers()[layer].feed_forward() else {
            continue;
        };
        for expert in 0..moe.router_spec().num_experts() {
            if group.placement.owner(layer, expert) != Some(owner) {
                continue;
            }
            let linear = |role: TensorRole, shape: [usize; 2]| {
                let binding =
                    resources
                        .experts()
                        .require_shape(layer, expert, role.clone(), &shape)?;
                PreparedLinear::from_parameter(
                    materializer.expert_parameter(layer, expert, binding)?,
                    role,
                )
            };
            experts.insert(
                ExpertId::new(layer, expert),
                Arc::new(PreparedSwiGlu::new(
                    linear(
                        TensorRole::RoutedExpertGate,
                        moe.expert().gate().weight_shape(),
                    )?,
                    linear(TensorRole::RoutedExpertUp, moe.expert().up().weight_shape())?,
                    linear(
                        TensorRole::RoutedExpertDown,
                        moe.expert().down().weight_shape(),
                    )?,
                    moe.expert().activation_limit(),
                )?),
            );
        }
    }
    Ok(LocalExperts(experts))
}

pub struct ExpertChild {
    identity: ProcessIdentity,
    boot: DecoderBoot,
    group: ExpertGroup,
    experts: LocalExperts,
    stats: ExpertProcessStats,
    width: usize,
    #[cfg(feature = "cuda")]
    cuda: Option<ferrule_model::transformer::CudaStandardDecoderOperators>,
}
impl ExpertChild {
    pub fn initialize(boot: ProcessBoot) -> std::result::Result<Self, ProcessHandlerError> {
        let config: ExpertBoot = serde_json::from_value(boot.config).map_err(startup)?;
        config.expert_boot.stage_boot().map_err(startup)?;
        let placement = config
            .expert_boot
            .experts
            .as_ref()
            .ok_or_else(|| startup("missing expert placement"))?;
        let slot = placement
            .members
            .iter()
            .position(|&r| r == config.owner)
            .ok_or_else(|| startup("unknown expert owner"))?;
        if boot.identity.rank.get() != config.owner {
            return Err(startup("expert Boot identity mismatch"));
        }
        let device = placement.devices[slot];
        let group = placement
            .decode(config.expert_boot.segment.layers.clone())
            .map_err(startup)?;
        let resources = config.expert_boot.load_resources().map_err(startup)?;
        let experts = prepare(
            &resources,
            &group,
            boot.identity.rank,
            config.expert_boot.kv.max_parameter_bytes,
        )
        .map_err(startup)?;
        #[allow(unused_mut)] // Mutated only by the CUDA initialization below.
        let mut child = super::decoder::Retained::new(Self {
            identity: boot.identity,
            width: resources.spec().hidden_size(),
            stats: ExpertProcessStats {
                pid: std::process::id(),
                owner: config.owner,
                device,
                owned_experts: experts.0.keys().map(|e| (e.layer, e.expert)).collect(),
                calls: 0,
                tokens: 0,
            },
            boot: config.expert_boot,
            group,
            experts,
            #[cfg(feature = "cuda")]
            cuda: None,
        });
        match device {
            DecoderDevice::Cpu => {}
            DecoderDevice::Cuda { ordinal } => {
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = ordinal;
                    return Err(startup(
                        "CUDA expert requires --features cuda; no CPU fallback",
                    ));
                }
                #[cfg(feature = "cuda")]
                {
                    let ops = std::rc::Rc::new(
                        ferrule_backend::cuda::operators::linear::CudaOperators::new_on_device(
                            ordinal,
                        )
                        .map_err(startup)?,
                    );
                    let parameters = child
                        .experts
                        .0
                        .values()
                        .flat_map(|e| {
                            [e.gate(), e.up(), e.down()].map(|l| l.parameter().binding().clone())
                        })
                        .collect::<Vec<_>>();
                    child.cuda = Some(
                        ferrule_model::transformer::CudaStandardDecoderOperators::new(
                            ops,
                            ferrule_model::execution::ExecutionPrecisionPolicy::f32(),
                            &parameters,
                        )
                        .map_err(startup)?,
                    );
                    let experts = child.experts.0.values().cloned().collect::<Vec<_>>();
                    for expert in experts {
                        let ops = child.cuda.as_mut().expect("owner operators");
                        let result = ops.prepare_expert(&expert);
                        if ops.needs_quarantine() {
                            std::mem::forget(result);
                            return Err(unknown("expert startup fence unknown"));
                        }
                        result.map_err(startup)?;
                    }
                }
            }
        }
        Ok(child.release())
    }
    fn compute(&mut self, tokens: &[ExpertToken]) -> Result<Vec<ExpertResult>> {
        #[cfg(feature = "cuda")]
        if let Some(ops) = self.cuda.as_mut() {
            let result = ferrule_model::transformer::CudaExpertWorker::new(
                self.identity.rank,
                &self.group.placement,
                &mut self.experts,
                ops,
            )
            .compute(tokens);
            if ops.needs_quarantine() {
                std::mem::forget(result);
                std::panic::panic_any(PanicQuiescence::Unknown);
            }
            return result;
        }
        CpuReferenceExpertWorker::new(self.identity.rank, &self.group.placement, &mut self.experts)
            .compute(tokens)
    }
}
fn startup(e: impl std::fmt::Display) -> ProcessHandlerError {
    ProcessHandlerError::fenced(ProcessFailureKind::Startup, e.to_string())
}
fn invalid(e: impl std::fmt::Display) -> ProcessHandlerError {
    ProcessHandlerError::fenced(ProcessFailureKind::CommandDecode, e.to_string())
}
fn unknown(e: impl std::fmt::Display) -> ProcessHandlerError {
    ProcessHandlerError::unknown(ProcessFailureKind::Handler, e.to_string())
}
impl ProcessChildHandler for ExpertChild {
    type Command = ExpertCommand;
    type Output = ExpertReply;
    fn execute(
        &mut self,
        request: ProcessRequest<ExpertCommand>,
    ) -> std::result::Result<ExpertReply, ProcessHandlerError> {
        if request.identity.owner != self.identity || request.session != 0 {
            return Err(invalid("expert envelope mismatch"));
        }
        let ExpertCommand::Compute {
            transaction,
            source,
            layer,
            tokens,
        } = request.command
        else {
            return Ok(ExpertReply::Stats(self.stats.clone()));
        };
        if transaction != request.identity.transaction.get()
            || source != self.group.source_rank.get()
            || !self.group.layers.contains(&layer)
            || tokens.len() > self.group.limits.max_tokens
        {
            return Err(invalid("expert context/bucket bounds mismatch"));
        }
        let mut ids = BTreeSet::new();
        let mut bytes = 0usize;
        for t in &tokens {
            bytes = t
                .values
                .len()
                .checked_mul(4)
                .and_then(|n| n.checked_add(bytes))
                .ok_or_else(|| invalid("expert byte overflow"))?;
            if bytes > self.group.limits.max_bytes
                || t.transaction != transaction
                || t.source != source
                || t.layer != layer
                || t.values.len() != self.width
                || t.row >= self.boot.kv.max_batch_tokens
                || t.route >= self.group.limits.max_tokens
                || self.group.placement.owner(t.layer, t.expert) != Some(self.identity.rank)
                || !ids.insert((t.row, t.route))
            {
                return Err(invalid("invalid expert token identity/placement/shape"));
            }
        }
        let tokens = tokens
            .into_iter()
            .map(ExpertTokenFrame::decode)
            .collect::<Result<Vec<_>>>()
            .map_err(invalid)?;
        let results = self.compute(&tokens).map_err(|e| {
            if matches!(self.stats.device, DecoderDevice::Cuda { .. }) {
                unknown(e)
            } else {
                ProcessHandlerError::fenced(ProcessFailureKind::Handler, e.to_string())
            }
        })?;
        let mut by_route = BTreeMap::new();
        for r in results {
            if r.owner_rank != self.identity.rank
                || r.transaction.get() != transaction
                || r.source_rank.get() != source
                || r.expert.layer != layer
                || by_route.insert((r.source_row, r.route_slot), r).is_some()
            {
                return Err(unknown("invalid expert computation identity"));
            }
        }
        let mut output = Vec::with_capacity(tokens.len());
        for token in &tokens {
            let r = by_route
                .remove(&(token.source_row, token.route_slot))
                .ok_or_else(|| unknown("missing expert computation result"))?;
            if r.sequence != token.sequence
                || r.expert != token.expert
                || r.output.len() != self.width
            {
                return Err(unknown("expert computation shape/identity mismatch"));
            }
            finite(&r.output).map_err(unknown)?;
            let mut frame = ExpertTokenFrame::encode(token);
            frame.values = r.output;
            output.push(frame);
        }
        if !by_route.is_empty() {
            return Err(unknown("extra expert result"));
        }
        self.stats.calls += 1;
        self.stats.tokens += tokens.len();
        Ok(ExpertReply::Results {
            owner: self.identity.rank.get(),
            tokens: output,
        })
    }
    fn shutdown(&mut self) -> std::result::Result<(), ProcessHandlerError> {
        #[cfg(feature = "cuda")]
        if let Some(ops) = self.cuda.as_mut() {
            ops.quiesce().map_err(unknown)?;
            if ops.needs_quarantine() {
                return Err(unknown("expert shutdown fence unknown"));
            }
        }
        Ok(())
    }
}

type Owner = ProcessRankOwner<ExpertBoot, ExpertCommand, ExpertReply>;
pub(super) struct ProcessExperts {
    pub group: ExpertGroup,
    owners: BTreeMap<u32, Owner>,
    stats: BTreeMap<u32, ExpertProcessStats>,
    unavailable: bool,
}
impl ProcessExperts {
    pub fn spawn(
        boot: &DecoderBoot,
        identity: ProcessIdentity,
        launch: ProcessLaunch,
        limits: ProcessFrameLimits,
    ) -> Result<Self> {
        let placement = boot
            .experts
            .as_ref()
            .ok_or_else(|| error("missing placement"))?;
        let mut this = Self {
            group: placement.decode(boot.segment.layers.clone())?,
            owners: BTreeMap::new(),
            stats: BTreeMap::new(),
            unavailable: false,
        };
        let timeout = Duration::from_millis(placement.timeout_ms);
        let options = ProcessOwnerConfig {
            frame_limits: limits,
            startup_timeout: timeout,
            command_timeout: timeout,
            ..ProcessOwnerConfig::default()
        };
        for &owner in &placement.members {
            let key = ProcessIdentity::new(
                identity.epoch,
                ParallelRankId::new(owner),
                ProcessOwnerInstanceId::new(u64::from(owner) + 1).map_err(error)?,
            );
            match Owner::spawn_inherited(
                launch.clone(),
                key,
                &ExpertBoot {
                    expert_boot: boot.clone(),
                    owner,
                },
                options,
            ) {
                Ok(child) => {
                    this.owners.insert(owner, child);
                }
                Err(e) => {
                    drop(e);
                    this.quarantine();
                    std::panic::panic_any(PanicQuiescence::Unknown);
                }
            }
        }
        this.process_stats()?;
        Ok(this)
    }
    fn quarantine(&mut self) {
        self.unavailable = true;
        self.owners.clear();
    }
    fn lost(&mut self, message: impl std::fmt::Display) -> ! {
        eprintln!("process expert quiescence unknown: {message}");
        self.quarantine();
        std::panic::panic_any(PanicQuiescence::Unknown)
    }
    pub fn process_stats(&mut self) -> Result<Vec<ExpertProcessStats>> {
        if self.unavailable {
            return Err(error("expert processes quarantined"));
        }
        for (&rank, owner) in &mut self.owners {
            let reply = owner.execute(ExecutionTransactionId::new(1)?, 0, &ExpertCommand::Stats);
            match reply {
                Ok(ExpertReply::Stats(stats))
                    if stats.owner == rank && Some(stats.pid) == owner.pid() =>
                {
                    self.stats.insert(rank, stats);
                }
                Err(e) if e.is_command_rejection() => return Err(error(e)),
                other => {
                    let msg = format!("invalid/lost expert statistics: {other:?}");
                    self.lost(msg);
                }
            }
        }
        Ok(self
            .group
            .members
            .iter()
            .map(|r| self.stats[&r.get()].clone())
            .collect())
    }
    pub fn shutdown(&mut self) -> Result<()> {
        if self.unavailable {
            return Err(error("expert shutdown cannot prove quiescence"));
        }
        let mut failures = Vec::new();
        for owner in self.owners.values_mut() {
            if let Err(e) = owner.shutdown() {
                failures.push(e.to_string());
            }
        }
        if !failures.is_empty() {
            self.lost(failures.join("; "));
        }
        self.owners.clear();
        Ok(())
    }
}
impl ferrule_model::transformer::expert_parallel::ExpertResultExecutor for ProcessExperts {
    fn execute(&mut self, bucket: ExpertTokenBucket) -> Result<Vec<ExpertResult>> {
        if self.unavailable {
            return Err(error("expert processes quarantined"));
        }
        let Some(first) = bucket.tokens.first() else {
            return Ok(Vec::new());
        };
        if bucket.tokens.len() > self.group.limits.max_tokens {
            return Err(error("expert token bound exceeded"));
        }
        let transaction = first.transaction;
        let command = ExpertCommand::Compute {
            transaction: transaction.get(),
            source: first.source_rank.get(),
            layer: first.expert.layer,
            tokens: bucket.tokens.iter().map(ExpertTokenFrame::encode).collect(),
        };
        let owner = self
            .owners
            .get_mut(&bucket.owner_rank.get())
            .ok_or_else(|| error("unknown expert process"))?;
        let reply = match owner.execute(transaction, 0, &command) {
            Ok(reply) => reply,
            Err(e) if e.is_command_rejection() => return Err(error(e)),
            Err(e) => self.lost(e),
        };
        let validated = (|| {
            let ExpertReply::Results { owner, tokens } = reply else {
                return Err(error("unexpected expert reply"));
            };
            if owner != bucket.owner_rank.get() || tokens.len() != bucket.tokens.len() {
                return Err(error("expert result count/owner mismatch"));
            }
            tokens
                .into_iter()
                .zip(&bucket.tokens)
                .map(|(frame, expected)| {
                    if frame.transaction != expected.transaction.get()
                        || frame.source != expected.source_rank.get()
                        || frame.sequence != expected.sequence
                        || frame.row != expected.source_row
                        || frame.route != expected.route_slot
                        || (frame.layer, frame.expert)
                            != (expected.expert.layer, expected.expert.expert)
                        || frame.weight.to_bits() != expected.weight.to_bits()
                        || frame.values.len() != expected.payload.len()
                    {
                        return Err(error("expert result identity/shape mismatch"));
                    }
                    finite(&frame.values)?;
                    Ok(ExpertResult::from_token(
                        expected,
                        bucket.owner_rank,
                        frame.values,
                    ))
                })
                .collect::<Result<Vec<_>>>()
        })();
        match validated {
            Ok(results) => Ok(results),
            Err(e) => self.lost(e),
        }
    }
}
