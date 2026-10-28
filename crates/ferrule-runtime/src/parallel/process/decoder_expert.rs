//! Independent PP -> EP pipes. No message is routed back through the root
//! while it waits for a stage. EP children inherit the PP isolation group, so
//! root timeout/quarantine terminates the entire tree, including stuck experts.

use std::collections::{BTreeMap, BTreeSet};

use std::sync::Arc;
use std::time::Duration;

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{ParallelRankId, Result};
use ferrule_model::moe::ExpertId;
use ferrule_model::transformer::{
    BoundDecoderResources, CpuReferenceExpertWorker, ExpertAvailability, ExpertDispatchContext,
    ExpertPlacementEpoch, ExpertProvider, ExpertResult, ExpertRouteResult, ExpertRouteToken,
    ExpertToken, ExpertTokenBucket, ExpertWorker, PreparedSwiGlu, StateDictMaterializer,
};
use serde::{Deserialize, Serialize};

use super::decoder::{DecoderBoot, DecoderDevice, ExpertRouteFrame, error, validate_expert_route};
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertRouteTokenFrame {
    pub route: ExpertRouteFrame,
    pub token: ExpertTokenFrame,
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
    fn encode_result(input: &ExpertToken, result: ExpertResult) -> Self {
        Self {
            transaction: result.transaction.get(),
            sequence: result.sequence,
            source: result.source_rank.get(),
            row: result.source_row,
            route: result.route_slot,
            layer: result.expert.layer,
            expert: result.expert.expert,
            weight: input.weight,
            values: result.output,
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

impl ExpertRouteTokenFrame {
    fn encode(route: &ExpertRouteToken) -> Self {
        Self {
            route: ExpertRouteFrame::encode(route.metadata),
            token: ExpertTokenFrame::encode(&route.token),
        }
    }

    fn encode_result(route: &ExpertRouteToken, result: ExpertRouteResult) -> Self {
        Self {
            route: ExpertRouteFrame::encode(result.metadata),
            token: ExpertTokenFrame::encode_result(&route.token, result.result),
        }
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
    /// Explicit route-aware seam. Legacy `Compute` is intentionally not
    /// upgraded in place: versioned/replicated placement must use this tag.
    ComputeRoutes {
        transaction: u64,
        source: u32,
        layer: usize,
        tokens: Vec<ExpertRouteTokenFrame>,
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
    /// Last completed activation call; echoes the existing outer transaction.
    pub last_context: Option<(u64, u32)>,
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
    RouteResults {
        owner: u32,
        tokens: Vec<ExpertRouteTokenFrame>,
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
    let catalog = group.catalog(resources)?;
    let materializer = StateDictMaterializer::new(max)?;
    let experts = catalog.prepare_selected(&materializer, |id| {
        group.placement.owner(id.layer, id.expert) == Some(owner)
    })?;
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
    pub fn initialize(
        boot: ProcessBoot,
        bind: impl FnOnce(&DecoderBoot) -> Result<BoundDecoderResources>,
    ) -> std::result::Result<Self, ProcessHandlerError> {
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
        if group.placement.is_replicated() {
            return Err(startup(format!(
                "unsupported replicated expert route placement: epoch={} owner={} slot/replica metadata has no process residency authority",
                group.placement.epoch().get(),
                config.owner
            )));
        }
        let resources = bind(&config.expert_boot).map_err(startup)?;
        config
            .expert_boot
            .validate_resources(&resources)
            .map_err(startup)?;
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
                last_context: None,
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
    fn compute_routes(&mut self, routes: &[ExpertRouteToken]) -> Result<Vec<ExpertRouteResult>> {
        #[cfg(feature = "cuda")]
        if let Some(ops) = self.cuda.as_mut() {
            let result = ferrule_model::transformer::CudaExpertWorker::new(
                self.identity.rank,
                &self.group.placement,
                &mut self.experts,
                ops,
            )
            .compute_routes(routes);
            if ops.needs_quarantine() {
                std::mem::forget(result);
                std::panic::panic_any(PanicQuiescence::Unknown);
            }
            return result;
        }
        CpuReferenceExpertWorker::new(self.identity.rank, &self.group.placement, &mut self.experts)
            .compute_routes(routes)
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

fn encode_legacy_results(
    tokens: &[ExpertToken],
    results: Vec<ExpertResult>,
    owner: ParallelRankId,
    width: usize,
) -> std::result::Result<Vec<ExpertTokenFrame>, ProcessHandlerError> {
    let mut by_route = BTreeMap::new();
    for result in results {
        if result.owner_rank != owner
            || result.output.len() != width
            || by_route
                .insert((result.source_row, result.route_slot), result)
                .is_some()
        {
            return Err(unknown("invalid expert computation identity"));
        }
    }
    let mut output = Vec::with_capacity(tokens.len());
    for token in tokens {
        let result = by_route
            .remove(&(token.source_row, token.route_slot))
            .ok_or_else(|| unknown("missing expert computation result"))?;
        if result.transaction != token.transaction
            || result.sequence != token.sequence
            || result.source_rank != token.source_rank
            || result.expert != token.expert
        {
            return Err(unknown("expert computation shape/identity mismatch"));
        }
        finite(&result.output).map_err(unknown)?;
        output.push(ExpertTokenFrame::encode_result(token, result));
    }
    if !by_route.is_empty() {
        return Err(unknown("extra expert result"));
    }
    Ok(output)
}

fn encode_route_results(
    routes: &[ExpertRouteToken],
    results: Vec<ExpertRouteResult>,
    owner: ParallelRankId,
    width: usize,
) -> std::result::Result<Vec<ExpertRouteTokenFrame>, ProcessHandlerError> {
    let mut by_route = BTreeMap::new();
    for result in results {
        if result.result.owner_rank != owner
            || result.result.output.len() != width
            || by_route
                .insert((result.result.source_row, result.result.route_slot), result)
                .is_some()
        {
            return Err(unknown("invalid expert route computation identity"));
        }
    }
    let mut output = Vec::with_capacity(routes.len());
    for route in routes {
        let result = by_route
            .remove(&(route.token.source_row, route.token.route_slot))
            .ok_or_else(|| unknown("missing expert route computation result"))?;
        if result.metadata != route.metadata
            || result.result.transaction != route.token.transaction
            || result.result.sequence != route.token.sequence
            || result.result.source_rank != route.token.source_rank
            || result.result.expert != route.token.expert
        {
            return Err(unknown("expert route computation identity mismatch"));
        }
        finite(&result.result.output).map_err(unknown)?;
        output.push(ExpertRouteTokenFrame::encode_result(route, result));
    }
    if !by_route.is_empty() {
        return Err(unknown("extra expert route result"));
    }
    Ok(output)
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
        enum Work {
            Legacy(Vec<ExpertTokenFrame>),
            Routes(Vec<ExpertRouteTokenFrame>),
        }
        let (transaction, source, layer, work) = match request.command {
            ExpertCommand::Stats => return Ok(ExpertReply::Stats(self.stats.clone())),
            ExpertCommand::Compute {
                transaction,
                source,
                layer,
                tokens,
            } => (transaction, source, layer, Work::Legacy(tokens)),
            ExpertCommand::ComputeRoutes {
                transaction,
                source,
                layer,
                tokens,
            } => (transaction, source, layer, Work::Routes(tokens)),
        };
        let (frame_count, source_mismatch) = match &work {
            Work::Legacy(tokens) => (
                tokens.len(),
                tokens.iter().any(|token| token.source != source),
            ),
            Work::Routes(tokens) => (
                tokens.len(),
                tokens.iter().any(|token| token.token.source != source),
            ),
        };
        if source != self.group.source_rank.get() || source_mismatch {
            let message = format!(
                "expert source identity mismatch: transaction={transaction} source={source} expected_source={}",
                self.group.source_rank.get()
            );
            return Err(invalid(message));
        }
        if transaction != request.identity.transaction.get() || !self.group.layers.contains(&layer)
        {
            return Err(invalid(format!(
                "expert context/transaction mismatch: transaction={} source={} layer={layer}",
                transaction, source
            )));
        }
        if frame_count > self.group.limits.max_tokens {
            return Err(invalid("expert token limit exceeded"));
        }
        let context = ExpertDispatchContext {
            transaction: ExecutionTransactionId::new(transaction).map_err(invalid)?,
            source_rank: ParallelRankId::new(source),
            layer,
        };
        let mut ids = BTreeSet::new();
        let mut bytes = 0usize;
        let mut check_frame =
            |t: &ExpertTokenFrame| -> std::result::Result<(), ProcessHandlerError> {
                bytes = bytes
                    .checked_add(
                        t.values
                            .len()
                            .checked_mul(4)
                            .ok_or_else(|| invalid("expert byte overflow"))?,
                    )
                    .ok_or_else(|| invalid("expert byte overflow"))?;
                if bytes > self.group.limits.max_bytes
                    || t.transaction != transaction
                    || t.source != source
                    || t.layer != layer
                    || t.values.len() != self.width
                    || t.row >= self.boot.kv.max_batch_tokens
                    || t.route >= self.group.limits.max_tokens
                    || !ids.insert((t.row, t.route))
                {
                    return Err(invalid("invalid expert token identity/shape"));
                }
                Ok(())
            };
        match work {
            Work::Legacy(frames) => {
                if self.group.placement.is_replicated()
                    || self.group.placement.epoch() != ExpertPlacementEpoch::INITIAL
                {
                    return Err(invalid(
                        "versioned or replicated placement requires route-aware expert command",
                    ));
                }
                for frame in &frames {
                    check_frame(frame)?;
                    if self.group.placement.owner(frame.layer, frame.expert)
                        != Some(self.identity.rank)
                    {
                        return Err(invalid(format!(
                            "expert route validation: legacy owner mismatch logical={}:{} owner={} receiver={}",
                            frame.layer,
                            frame.expert,
                            self.identity.rank.get(),
                            self.identity.rank.get()
                        )));
                    }
                }
                let tokens = frames
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
                let output =
                    encode_legacy_results(&tokens, results, self.identity.rank, self.width)?;
                self.stats.last_context = Some((transaction, source));
                self.stats.calls += 1;
                self.stats.tokens += tokens.len();
                Ok(ExpertReply::Results {
                    owner: self.identity.rank.get(),
                    tokens: output,
                })
            }
            Work::Routes(frames) => {
                let mut route_tokens = Vec::with_capacity(frames.len());
                for frame in frames {
                    check_frame(&frame.token)?;
                    let token_expert =
                        ferrule_model::moe::ExpertId::new(frame.token.layer, frame.token.expert);
                    let metadata = validate_expert_route(
                        &self.group.placement,
                        self.identity.rank,
                        context,
                        token_expert,
                        frame.route,
                    )
                    .map_err(invalid)?;
                    route_tokens.push(ExpertRouteToken {
                        metadata,
                        token: frame.token.decode().map_err(invalid)?,
                    });
                }
                let results = self.compute_routes(&route_tokens).map_err(|e| {
                    if matches!(self.stats.device, DecoderDevice::Cuda { .. }) {
                        unknown(e)
                    } else {
                        ProcessHandlerError::fenced(ProcessFailureKind::Handler, e.to_string())
                    }
                })?;
                let output =
                    encode_route_results(&route_tokens, results, self.identity.rank, self.width)?;
                self.stats.last_context = Some((transaction, source));
                self.stats.calls += 1;
                self.stats.tokens += route_tokens.len();
                Ok(ExpertReply::RouteResults {
                    owner: self.identity.rank.get(),
                    tokens: output,
                })
            }
        }
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

fn validate_route_reply_values(
    values: &[f32],
    context: ExpertDispatchContext,
    metadata: ferrule_model::transformer::ExpertRouteMetadata,
) -> Result<()> {
    finite(values).map_err(|source| {
        error(format!(
            "nonfinite expert route reply: transaction={} source={} logical={}:{} slot={} replica={} owner={} epoch={} reason={source}",
            context.transaction.get(),
            context.source_rank.get(),
            metadata.logical_expert.layer,
            metadata.logical_expert.expert,
            metadata.physical_slot,
            metadata.replica,
            metadata.owner_rank.get(),
            metadata.placement_epoch.get()
        ))
    })
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
        boot.stage_boot()?;
        if identity.rank.get() != boot.rank {
            return Err(error("expert caller Boot identity mismatch"));
        }
        let placement = boot
            .experts
            .as_ref()
            .ok_or_else(|| error("missing placement"))?;
        let group = placement.decode(boot.segment.layers.clone())?;
        if group.placement.is_replicated() {
            return Err(error(format!(
                "unsupported replicated process expert route: epoch={} source={} owner/slot/replica residency authority is unavailable",
                group.placement.epoch().get(),
                group.source_rank.get()
            )));
        }
        let mut this = Self {
            group,
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

    /// Route-aware production seam used by versioned process stages. The model
    /// plan authenticates route membership; this method preserves that exact
    /// metadata through the owner pipe instead of rebuilding a logical primary.
    pub(super) fn execute_routes(
        &mut self,
        context: ExpertDispatchContext,
        routes: &[ExpertRouteToken],
    ) -> Result<Vec<ExpertRouteResult>> {
        if self.unavailable {
            return Err(error("expert processes quarantined"));
        }
        let Some(first) = routes.first() else {
            return Ok(Vec::new());
        };
        let owner = first.metadata.owner_rank;
        self.group.validate_context(context).map_err(|_| {
            error(format!(
                "expert route validation: transaction={} source={} layer={} does not match the registered process expert group",
                context.transaction.get(),
                context.source_rank.get(),
                context.layer
            ))
        })?;
        if first.token.transaction != context.transaction
            || first.token.source_rank != context.source_rank
            || first.token.expert.layer != context.layer
        {
            return Err(error(format!(
                "expert route validation: transaction={} source={} layer={} first route context mismatch",
                context.transaction.get(),
                context.source_rank.get(),
                context.layer
            )));
        }
        if routes.len() > self.group.limits.max_tokens {
            return Err(error(format!(
                "expert route validation: transaction={} source={} owner={} route count exceeds limit",
                context.transaction.get(),
                context.source_rank.get(),
                owner.get()
            )));
        }
        for route in routes {
            if route.metadata.owner_rank != owner
                || route.token.transaction != context.transaction
                || route.token.source_rank != context.source_rank
                || route.token.expert.layer != context.layer
            {
                return Err(error(format!(
                    "expert route validation: transaction={} source={} owner={} logical={}:{} slot={} replica={} route bucket mismatch",
                    context.transaction.get(),
                    context.source_rank.get(),
                    owner.get(),
                    route.token.expert.layer,
                    route.token.expert.expert,
                    route.metadata.physical_slot,
                    route.metadata.replica
                )));
            }
            validate_expert_route(
                &self.group.placement,
                owner,
                context,
                route.token.expert,
                ExpertRouteFrame::encode(route.metadata),
            )?;
        }
        let command = ExpertCommand::ComputeRoutes {
            transaction: context.transaction.get(),
            source: context.source_rank.get(),
            layer: context.layer,
            tokens: routes.iter().map(ExpertRouteTokenFrame::encode).collect(),
        };
        let child = self
            .owners
            .get_mut(&owner.get())
            .ok_or_else(|| error(format!("unknown expert process owner={}", owner.get())))?;
        let reply = match child.execute(context.transaction, 0, &command) {
            Ok(reply) => reply,
            Err(e) if e.is_command_rejection() => return Err(error(e)),
            Err(e) => self.lost(e),
        };
        let ExpertReply::RouteResults {
            owner: actual_owner,
            tokens,
        } = reply
        else {
            self.lost("unexpected non-route expert reply");
        };
        if actual_owner != owner.get() || tokens.len() != routes.len() {
            self.lost(format!(
                "expert route result owner/count mismatch: transaction={} source={} expected_owner={} actual_owner={} expected_count={} actual_count={}",
                context.transaction.get(),
                context.source_rank.get(),
                owner.get(),
                actual_owner,
                routes.len(),
                tokens.len()
            ));
        }
        let mut results = Vec::with_capacity(tokens.len());
        for (frame, expected) in tokens.into_iter().zip(routes) {
            if frame.route.metadata() != expected.metadata
                || frame.token.transaction != expected.token.transaction.get()
                || frame.token.source != expected.token.source_rank.get()
                || frame.token.sequence != expected.token.sequence
                || frame.token.row != expected.token.source_row
                || frame.token.route != expected.token.route_slot
                || (frame.token.layer, frame.token.expert)
                    != (expected.token.expert.layer, expected.token.expert.expert)
                || frame.token.weight.to_bits() != expected.token.weight.to_bits()
                || frame.token.values.len() != expected.token.payload.len()
            {
                self.lost(format!(
                    "expert route result identity mismatch: transaction={} source={} logical={}:{} slot={} replica={} owner={}",
                    context.transaction.get(),
                    context.source_rank.get(),
                    expected.token.expert.layer,
                    expected.token.expert.expert,
                    expected.metadata.physical_slot,
                    expected.metadata.replica,
                    owner.get()
                ));
            }
            if let Err(source) =
                validate_route_reply_values(&frame.token.values, context, expected.metadata)
            {
                self.lost(source);
            }
            results.push(ExpertRouteResult {
                metadata: expected.metadata,
                result: ExpertResult::from_token(&expected.token, owner, frame.token.values),
            });
        }
        Ok(results)
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
        if bucket
            .tokens
            .iter()
            .any(|token| token.source_rank != self.group.source_rank)
        {
            self.lost("expert dispatch source differs from the booted PP caller");
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

#[cfg(test)]
#[path = "../../../tests/unit/parallel/expert_semantics.rs"]
mod semantic_tests;
