//! Scoped numeric expert processes. The existing DP tickets own dispatch and
//! ProcessRankOwner owns pipes/isolation/reaping; neither owns decoder KV.
//!
//! # Public composition boundaries
//!
//! This module is available on Unix. Use `NumericExpertChild` through this
//! module; its private `child` module selects the CUDA implementation or the
//! unavailable CPU facade. `initialize` has the same signature in both builds
//! and validates Boot metadata before invoking a binder or rejecting CUDA.
//! `initialize_bound` is available only with the `cuda` feature.
//!
//! `NumericFrameSize` and `FrameSizeOverflow` are the public sizing preflight.
//! Do not import the private `frame_budget` or IPC codec modules. Wire types,
//! including `NumericProcessStats`, remain available on Unix CPU builds.
#[cfg(feature = "cuda")]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::topology::ExpertSourceScope;
use ferrule_common::{Error, ParallelRankId, Result};
#[cfg(any(feature = "cuda", test))]
use ferrule_model::transformer::ExpertTokenBucket;
#[cfg(feature = "cuda")]
use ferrule_model::transformer::expert_parallel::ExpertRouteToken;
use ferrule_model::transformer::expert_parallel::{ExpertDispatchContext, ExpertRouteResult};
use ferrule_model::transformer::host_experts::{
    ExpertPrewarmMode, HostExpertCacheOptions, HostMemoryBudget,
};
use ferrule_model::transformer::{
    BoundDecoderResources, ExpertResult, ExpertToken, NumericFp8Precision,
};
use serde::{Deserialize, Serialize};

use super::data::{
    DataParallelConfig, DataParallelExecutor, PanicQuiescence, ReplicaWorker, WorkRequest,
};
use super::expert::{
    Command, ExpertGroup, ExpertParallelExecutor, ExpertRank, ExpertRankWorkers, Reply,
};
/// Numeric batch preflight only; the process codec remains private.
#[doc(inline)]
pub use super::frame_budget::{FrameSizeOverflow, NumericFrameSize};
use super::process::decoder::{
    DecoderDevice, ExpertPlacementFrame, ExpertRouteFrame, validate_expert_route,
};
use super::process::*;

#[cfg(feature = "cuda")]
#[path = "expert_process/child_cuda.rs"]
mod child;
#[cfg(not(feature = "cuda"))]
#[path = "expert_process/child_unavailable.rs"]
mod child;

pub const NUMERIC_EXPERT_WIRE_VERSION: u16 = 2;
fn error(message: impl std::fmt::Display) -> Error {
    Error::Execution {
        message: format!("numeric process EP: {message}"),
    }
}
fn startup(message: impl std::fmt::Display) -> ProcessHandlerError {
    ProcessHandlerError::fenced(ProcessFailureKind::Startup, message.to_string())
}
#[cfg(feature = "cuda")]
fn invalid(message: impl std::fmt::Display) -> ProcessHandlerError {
    ProcessHandlerError::fenced(ProcessFailureKind::CommandDecode, message.to_string())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericSource {
    pub layer: usize,
    pub expert: usize,
    pub identity: [u8; 32],
    pub content_hash: [u8; 32],
    pub encoding: u32,
    pub generation: u64,
    pub bytes: u64,
}

/// Source fields describe the existing catalog, not a new hash of weight bytes.
pub fn owned_catalog(
    resources: &BoundDecoderResources,
    group: &ExpertGroup,
    owner: ParallelRankId,
) -> Result<Vec<NumericSource>> {
    group.validate_resources(resources)?;
    if !group.members.contains(&owner) {
        return Err(error("unknown catalog owner"));
    }
    let mut result = Vec::new();
    for entry in resources.source_catalogs().expert_materialization().iter() {
        let Some((layer, expert)) = entry.resource().routed_expert_coordinates() else {
            continue;
        };
        let (layer, expert) = (layer.get() as usize, expert.get() as usize);
        if group.layers.contains(&layer) && group.placement.owner(layer, expert) == Some(owner) {
            let source = entry.source();
            result.push(NumericSource {
                layer,
                expert,
                identity: source.identity().bytes(),
                content_hash: source.content_hash().bytes(),
                encoding: source.encoding().get(),
                generation: source.generation().get(),
                bytes: entry.storage_bytes(),
            });
        }
    }
    result.sort_by_key(|source| (source.layer, source.expert));
    if result.is_empty() {
        return Err(error("empty owned catalog"));
    }
    Ok(result)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericBoot {
    pub numeric_expert_boot: u16,
    /// Omitted legacy v2 boots retain F32; unknown profiles fail deserialization.
    /// Rust constructors use `NumericFp8Precision::expert_wire_default()`, not
    /// the model enum's historical BF16 default.
    #[serde(
        default = "NumericFp8Precision::expert_wire_default",
        serialize_with = "NumericFp8Precision::serialize_expert_wire",
        deserialize_with = "NumericFp8Precision::deserialize_expert_wire"
    )]
    pub numeric_profile: NumericFp8Precision,
    pub checkpoint: PathBuf,
    pub total_layers: usize,
    pub layers: std::ops::Range<usize>,
    pub placement: ExpertPlacementFrame,
    pub owner: u32,
    pub hidden: usize,
    pub max_rows: usize,
    pub max_parameter_bytes: u64,
    pub host_max_bytes: u64,
    pub host_max_experts: usize,
    pub host_workers: usize,
    pub host_headroom_bytes: u64,
    pub device_bytes: usize,
    pub workspace_bytes: usize,
    pub expected: Vec<NumericSource>,
}
impl NumericBoot {
    pub fn group(&self) -> Result<ExpertGroup> {
        #[cfg(test)]
        boot_cache_tests::record_group_validation();
        super::expert::validate_numeric_profile(self.numeric_profile)?;
        if self.numeric_expert_boot != NUMERIC_EXPERT_WIRE_VERSION
            || self.total_layers == 0
            || self.layers.is_empty()
            || self.layers.end > self.total_layers
            || self.placement.source_scope != ExpertSourceScope::ExternalStage
            || self.placement.devices.len() != self.placement.members.len()
            || self.checkpoint.as_os_str().is_empty()
            || self.hidden == 0
            || self.max_rows == 0
            || self.max_parameter_bytes == 0
            || self.host_max_bytes == 0
            || self.host_max_experts == 0
            || self.workspace_bytes == 0
            || self.workspace_bytes >= self.device_bytes
        {
            return Err(error("invalid numeric v2 boot scope/budget"));
        }
        let group = self.placement.decode(self.layers.clone())?;
        group.validate()?;
        if let Some(entry) = group.placement.entries().find(|entry| entry.replica != 0) {
            return Err(error(format!(
                "numeric process route unsupported: placement_epoch={} logical={}:{} physical_slot={} replica={} owner={} has no explicit residency/provider authority",
                group.placement.epoch().get(),
                entry.logical_expert.layer,
                entry.logical_expert.expert,
                entry.physical_slot,
                entry.replica,
                entry.owner_rank.get()
            )));
        }
        if self.placement.members.contains(&self.placement.source)
            || self.placement.devices.iter().any(|device| !matches!(device, DecoderDevice::Cuda { ordinal } if i32::try_from(*ordinal).is_ok()))
            || !group.members.contains(&ParallelRankId::new(self.owner))
            || Duration::from_millis(self.placement.timeout_ms).is_zero()
            || std::time::Instant::now().checked_add(Duration::from_millis(self.placement.timeout_ms)).is_none()
        { return Err(error("invalid numeric owner/source/device/deadline")); }
        self.host_options().validate()?;
        let declared = group
            .placement
            .entries()
            .filter(|entry| entry.owner_rank.get() == self.owner)
            .map(|entry| (entry.logical_expert.layer, entry.logical_expert.expert))
            .collect::<BTreeSet<_>>();
        let actual = self
            .expected
            .iter()
            .map(|s| (s.layer, s.expert))
            .collect::<BTreeSet<_>>();
        if declared.is_empty()
            || declared != actual
            || actual.len() != self.expected.len()
            || actual.len() > self.host_max_experts
            || self
                .expected
                .windows(2)
                .any(|w| (w[0].layer, w[0].expert) >= (w[1].layer, w[1].expert))
            || self.expected.iter().any(|s| {
                s.identity == [0; 32]
                    || s.content_hash == [0; 32]
                    || s.encoding == 0
                    || s.generation == 0
                    || s.bytes == 0
            })
        {
            return Err(error("invalid exact owned catalog"));
        }
        let payload = self.expected.iter().try_fold(0u64, |n, s| {
            n.checked_add(s.bytes)
                .ok_or_else(|| error("payload overflow"))
        })?;
        if payload > self.host_max_bytes || payload > self.device_bytes as u64 {
            return Err(error("owned payload exceeds budget"));
        }
        Ok(group)
    }
    /// Conservative integer-JSON bounds for one routed bucket. Factories may
    /// enlarge payload limits within the existing process frame ceiling; they
    /// must not start children whose legal batch can never fit the wire.
    pub fn validate_frame_limits(&self, limits: ProcessFrameLimits) -> Result<()> {
        limits.validate().map_err(error)?;
        let size =
            NumericFrameSize::new(self.hidden, self.placement.max_tokens).map_err(|overflow| {
                match overflow {
                    FrameSizeOverflow::Payload => error("numeric frame size overflow"),
                    FrameSizeOverflow::Envelope => error("numeric envelope overflow"),
                }
            })?;
        if !size.fits(limits) {
            return Err(error("numeric routed batch exceeds process frame limits"));
        }
        Ok(())
    }

    pub fn rank(&self) -> Result<ExpertRank> {
        self.group()?;
        let slot = self
            .placement
            .members
            .iter()
            .position(|&m| m == self.owner)
            .ok_or_else(|| error("missing owner"))?;
        Ok(ExpertRank {
            owner: ParallelRankId::new(self.owner),
            slot: ParallelRankId::new(slot as u32),
        })
    }
    pub fn device(&self) -> Result<DecoderDevice> {
        Ok(self.placement.devices[self.rank()?.slot.get() as usize])
    }
    pub fn host_options(&self) -> HostExpertCacheOptions {
        HostExpertCacheOptions {
            mode: ExpertPrewarmMode::Full,
            max_experts: self.host_max_experts,
            max_bytes: self.host_max_bytes,
            workers: self.host_workers,
            memory_budget: HostMemoryBudget {
                headroom_bytes: self.host_headroom_bytes,
                ..Default::default()
            },
        }
    }
    pub fn validate_resources(&self, resources: &BoundDecoderResources) -> Result<()> {
        let group = self.group()?;
        if resources.spec().layers().len() != self.total_layers
            || resources.spec().hidden_size() != self.hidden
            || owned_catalog(resources, &group, ParallelRankId::new(self.owner))? != self.expected
        {
            return Err(error("rebound source catalog/geometry differs from boot"));
        }
        Ok(())
    }
}

/// Immutable, owner-private startup authority. A new owner/config always builds
/// a new token; no process-global cache or mutable Boot alias survives startup.
struct ValidatedNumericBoot {
    boot: NumericBoot,
    identity: ProcessIdentity,
    group: ExpertGroup,
    rank: ExpertRank,
    device: DecoderDevice,
    payload_bytes: u64,
}

impl ValidatedNumericBoot {
    fn build(
        boot: NumericBoot,
        identity: ProcessIdentity,
        limits: ProcessFrameLimits,
    ) -> Result<Arc<Self>> {
        let group = boot.group()?;
        boot.validate_frame_limits(limits)?;
        if identity.rank.get() != boot.owner {
            return Err(error("numeric envelope owner mismatch"));
        }
        let rank = rank_for_group(&boot, &group)?;
        let device = boot.placement.devices[rank.slot.get() as usize];
        let payload_bytes = boot.expected.iter().map(|source| source.bytes).sum();
        tracing::debug!(
            owner = identity.rank.get(),
            epoch = identity.epoch.get(),
            slot = rank.slot.get(),
            numeric_profile = ?boot.numeric_profile,
            workspace_reservation_bytes = boot.workspace_bytes,
            "validated numeric process boot"
        );
        Ok(Arc::new(Self {
            boot,
            identity,
            group,
            rank,
            device,
            payload_bytes,
        }))
    }

    #[cfg(any(feature = "cuda", test))]
    fn validate_resources(&self, resources: &BoundDecoderResources) -> Result<()> {
        if resources.spec().layers().len() != self.boot.total_layers
            || resources.spec().hidden_size() != self.boot.hidden
            || owned_catalog(resources, &self.group, self.rank.owner)? != self.boot.expected
        {
            return Err(error("rebound source catalog/geometry differs from boot"));
        }
        Ok(())
    }

    fn validate_token_frames(
        &self,
        context: ExpertDispatchContext,
        frames: &[NumericToken],
    ) -> Result<()> {
        validate_frame_bounds(&self.boot, frames)?;
        validate_frame_metadata(
            &self.group,
            self.rank.owner,
            context,
            frames.iter().map(|frame| {
                (
                    frame.transaction,
                    ParallelRankId::new(frame.source),
                    frame.layer,
                    frame.expert,
                    frame.values.len(),
                )
            }),
        )
    }

    fn validate_route_frames(
        &self,
        context: ExpertDispatchContext,
        frames: &[NumericRouteToken],
    ) -> Result<()> {
        if frames.len() > self.boot.placement.max_tokens {
            return Err(error("numeric route token limit"));
        }
        let mut seen = BTreeSet::new();
        for frame in frames {
            validate_frame_bounds(&self.boot, std::slice::from_ref(&frame.token))?;
            let token_expert =
                ferrule_model::moe::ExpertId::new(frame.token.layer, frame.token.expert);
            validate_expert_route(
                &self.group.placement,
                self.rank.owner,
                context,
                token_expert,
                frame.route,
            )?;
            if !seen.insert((frame.token.row, frame.token.route)) {
                return Err(error("duplicate numeric route identity"));
            }
        }
        validate_frame_metadata(
            &self.group,
            self.rank.owner,
            context,
            frames.iter().map(|frame| {
                (
                    frame.token.transaction,
                    ParallelRankId::new(frame.token.source),
                    frame.token.layer,
                    frame.token.expert,
                    frame.token.values.len(),
                )
            }),
        )
    }

    #[cfg(any(feature = "cuda", test))]
    fn validate_tokens(
        &self,
        context: ExpertDispatchContext,
        frames: Vec<NumericToken>,
    ) -> Result<Vec<ExpertToken>> {
        let bucket = decode_bucket(&self.boot, frames)?;
        self.group.validate_bucket(context, &bucket)?;
        Ok(bucket.tokens)
    }
}

fn rank_for_group(boot: &NumericBoot, group: &ExpertGroup) -> Result<ExpertRank> {
    let slot = group
        .members
        .iter()
        .position(|&member| member.get() == boot.owner)
        .ok_or_else(|| error("missing validated numeric owner"))?;
    Ok(ExpertRank {
        owner: ParallelRankId::new(boot.owner),
        slot: ParallelRankId::new(slot as u32),
    })
}

/// F32 bit patterns avoid JSON decimal roundtrips and preserve signed zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericToken {
    pub transaction: u64,
    pub sequence: u64,
    pub source: u32,
    pub row: usize,
    pub route: usize,
    pub layer: usize,
    pub expert: usize,
    pub weight: u32,
    pub values: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericRouteToken {
    pub route: ExpertRouteFrame,
    pub token: NumericToken,
}
impl NumericToken {
    fn encode(t: &ExpertToken) -> Self {
        Self::encode_values(t, t.payload.iter().map(|v| v.to_bits()).collect())
    }
    fn encode_values(t: &ExpertToken, values: Vec<u32>) -> Self {
        Self {
            transaction: t.transaction.get(),
            sequence: t.sequence,
            source: t.source_rank.get(),
            row: t.source_row,
            route: t.route_slot,
            layer: t.expert.layer,
            expert: t.expert.expert,
            weight: t.weight.to_bits(),
            values,
        }
    }
    fn matches_input(&self, expected: &ExpertToken, hidden: usize) -> bool {
        self.transaction == expected.transaction.get()
            && self.sequence == expected.sequence
            && self.source == expected.source_rank.get()
            && self.row == expected.source_row
            && self.route == expected.route_slot
            && self.layer == expected.expert.layer
            && self.expert == expected.expert.expert
            && self.weight == expected.weight.to_bits()
            && self.values.len() == hidden
    }
    fn decode(self) -> Result<ExpertToken> {
        if !f32::from_bits(self.weight).is_finite()
            || self.values.iter().any(|&v| !f32::from_bits(v).is_finite())
        {
            return Err(error("nonfinite numeric token"));
        }
        Ok(ExpertToken {
            transaction: ExecutionTransactionId::new(self.transaction)?,
            sequence: self.sequence,
            source_rank: ParallelRankId::new(self.source),
            source_row: self.row,
            route_slot: self.route,
            expert: ferrule_model::moe::ExpertId::new(self.layer, self.expert),
            weight: f32::from_bits(self.weight),
            payload: self.values.into_iter().map(f32::from_bits).collect(),
        })
    }
}

#[cfg(feature = "cuda")]
impl NumericRouteToken {
    fn encode_result(result: &ExpertRouteResult, weight: u32) -> Self {
        Self {
            route: ExpertRouteFrame::encode(result.metadata),
            token: NumericToken {
                transaction: result.result.transaction.get(),
                sequence: result.result.sequence,
                source: result.result.source_rank.get(),
                row: result.result.source_row,
                route: result.result.route_slot,
                layer: result.result.expert.layer,
                expert: result.result.expert.expert,
                weight,
                values: result.result.output.iter().map(|v| v.to_bits()).collect(),
            },
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum NumericCommand {
    Stats,
    Compute {
        transaction: u64,
        source: u32,
        layer: usize,
        tokens: Vec<NumericToken>,
    },
    /// Explicit route-aware process command. Legacy `Compute` remains fixed
    /// single-replica only and cannot select a logical primary implicitly.
    ComputeRoutes {
        transaction: u64,
        source: u32,
        layer: usize,
        tokens: Vec<NumericRouteToken>,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericProcessStats {
    pub pid: u32,
    pub owner: u32,
    pub slot: u32,
    pub device: DecoderDevice,
    pub owned_experts: Vec<(usize, usize)>,
    pub calls: usize,
    pub tokens: usize,
    pub host_payload_bytes: u64,
    pub host_read_bytes: u64,
    pub host_hits: u64,
    pub resident_experts: usize,
    pub resident_bytes: usize,
    pub uploads: u64,
    pub evictions: u64,
    pub pending_upload_bytes: usize,
    pub quarantined: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum NumericReply {
    Stats(NumericProcessStats),
    Results {
        owner: u32,
        tokens: Vec<NumericToken>,
    },
    RouteResults {
        owner: u32,
        tokens: Vec<NumericRouteToken>,
    },
}

/// Stable Unix endpoint path for both CUDA and unavailable CPU builds.
#[doc(inline)]
pub use child::NumericExpertChild;

impl NumericExpertChild {
    /// Validate the complete metadata-only Boot before binding or rejecting CUDA.
    ///
    /// The application binder returns `ferrule_common::Result<BoundDecoderResources>`;
    /// this endpoint returns `Result<Self, ProcessHandlerError>`. Without CUDA,
    /// the binder is never invoked and no checkpoint payload is read.
    pub fn initialize(
        boot: ProcessBoot,
        bind: impl FnOnce(&NumericBoot) -> Result<BoundDecoderResources>,
    ) -> std::result::Result<Self, ProcessHandlerError> {
        let validated = ValidatedNumericBoot::build(
            serde_json::from_value(boot.config).map_err(startup)?,
            boot.identity,
            boot.limits,
        )
        .map_err(startup)?;
        Self::initialize_checked(validated, bind)
    }
}

#[cfg(any(feature = "cuda", test))]
fn preflight_numeric_source(
    host: &ferrule_model::transformer::host_experts::HostExpertCache,
    generation: u64,
) -> Result<()> {
    if host.generation() != generation {
        return Err(error("numeric host generation changed after Ready"));
    }
    host.preflight()
}

#[cfg(test)]
fn validate_tokens(
    boot: &NumericBoot,
    context: ExpertDispatchContext,
    frames: Vec<NumericToken>,
) -> Result<Vec<ExpertToken>> {
    let bucket = decode_bucket(boot, frames)?;
    boot.group()?.validate_bucket(context, &bucket)?;
    Ok(bucket.tokens)
}

#[cfg(any(feature = "cuda", test))]
fn decode_bucket(boot: &NumericBoot, frames: Vec<NumericToken>) -> Result<ExpertTokenBucket> {
    if frames.len() > boot.placement.max_tokens {
        return Err(error("numeric token limit"));
    }
    let tokens = frames
        .into_iter()
        .map(NumericToken::decode)
        .collect::<Result<Vec<_>>>()?;
    let mut seen = BTreeSet::new();
    for t in &tokens {
        if t.payload.len() != boot.hidden
            || t.source_row >= boot.max_rows
            || t.route_slot >= boot.placement.max_tokens
            || !seen.insert((t.source_row, t.route_slot))
        {
            return Err(error("numeric route/width bounds"));
        }
    }
    Ok(ExpertTokenBucket {
        owner_rank: ParallelRankId::new(boot.owner),
        tokens,
    })
}

#[cfg(test)]
fn validate_token_frames(
    boot: &NumericBoot,
    context: ExpertDispatchContext,
    frames: &[NumericToken],
) -> Result<()> {
    validate_frame_bounds(boot, frames)?;
    validate_frame_metadata(
        &boot.group()?,
        ParallelRankId::new(boot.owner),
        context,
        frames.iter().map(|frame| {
            (
                frame.transaction,
                ParallelRankId::new(frame.source),
                frame.layer,
                frame.expert,
                frame.values.len(),
            )
        }),
    )
}

// Preserve finite/transaction checks before route/group errors without decoding
// or copying activation vectors on the proxy.
fn validate_frame_bounds(boot: &NumericBoot, frames: &[NumericToken]) -> Result<()> {
    if frames.len() > boot.placement.max_tokens {
        return Err(error("numeric token limit"));
    }
    for frame in frames {
        if !f32::from_bits(frame.weight).is_finite()
            || frame.values.iter().any(|&v| !f32::from_bits(v).is_finite())
        {
            return Err(error("nonfinite numeric token"));
        }
        ExecutionTransactionId::new(frame.transaction)?;
    }
    let mut seen = BTreeSet::new();
    for frame in frames {
        if frame.values.len() != boot.hidden
            || frame.row >= boot.max_rows
            || frame.route >= boot.placement.max_tokens
            || !seen.insert((frame.row, frame.route))
        {
            return Err(error("numeric route/width bounds"));
        }
    }
    Ok(())
}

fn validate_frame_metadata<I>(
    group: &ExpertGroup,
    owner: ParallelRankId,
    context: ExpertDispatchContext,
    frames: I,
) -> Result<()>
where
    I: ExactSizeIterator<Item = (u64, ParallelRankId, usize, usize, usize)>,
{
    group.validate_token_metadata(context, owner, frames)
}

#[cfg(feature = "cuda")]
fn encode_numeric_route_results(
    routes: &[ExpertRouteToken],
    results: Vec<ExpertRouteResult>,
    owner: ParallelRankId,
) -> Result<Vec<NumericRouteToken>> {
    let mut by_route = BTreeMap::new();
    for result in results {
        if result.result.owner_rank != owner
            || by_route
                .insert((result.result.source_row, result.result.route_slot), result)
                .is_some()
        {
            return Err(error("invalid numeric route computation identity"));
        }
    }
    let mut output = Vec::with_capacity(routes.len());
    for route in routes {
        let result = by_route
            .remove(&(route.token.source_row, route.token.route_slot))
            .ok_or_else(|| error("missing numeric route computation result"))?;
        if result.metadata != route.metadata
            || result.result.transaction != route.token.transaction
            || result.result.sequence != route.token.sequence
            || result.result.source_rank != route.token.source_rank
            || result.result.expert != route.token.expert
        {
            return Err(error("numeric route computation identity mismatch"));
        }
        if result.result.output.iter().any(|value| !value.is_finite()) {
            return Err(error("nonfinite numeric route result"));
        }
        output.push(NumericRouteToken::encode_result(
            &result,
            route.token.weight.to_bits(),
        ));
    }
    if !by_route.is_empty() {
        return Err(error("extra numeric route computation result"));
    }
    Ok(output)
}

type Owner = ProcessRankOwner<NumericBoot, NumericCommand, NumericReply>;
struct Control {
    owner: Mutex<Option<Owner>>,
    stop: AtomicBool,
    unknown: AtomicBool,
}
impl Control {
    fn terminate(&self) {
        self.stop.store(true, Ordering::Release);
        let mut owner = self.owner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(mut owner) = owner.take() {
            owner.terminate();
        }
    }
}
#[derive(Clone)]
pub(super) struct ProcessControls(Arc<Vec<Arc<Control>>>);
impl ProcessControls {
    pub(super) fn terminate(&self) {
        for c in self.0.iter() {
            c.stop.store(true, Ordering::Release);
        }
        for c in self.0.iter() {
            c.terminate();
        }
    }
}
struct Proxy {
    control: Arc<Control>,
    validated: Arc<ValidatedNumericBoot>,
    pid: u32,
    ready: NumericProcessStats,
}
impl Proxy {
    fn call(
        &self,
        transaction: ExecutionTransactionId,
        command: &NumericCommand,
    ) -> Result<NumericReply> {
        let control = &self.control;
        let result = {
            let mut owner = control.owner.lock().unwrap_or_else(|e| e.into_inner());
            let Some(owner) = owner.as_mut() else {
                std::panic::panic_any(PanicQuiescence::Unknown)
            };
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if owner.identity() != self.validated.identity || owner.pid() != Some(self.pid) {
                    std::panic::panic_any(PanicQuiescence::Unknown);
                }
                owner.execute_observed(transaction, 0, command, &mut || {
                    if control.stop.load(Ordering::Acquire) {
                        std::panic::panic_any(PanicQuiescence::Unknown);
                    }
                })
            }))
        };
        match result {
            Ok(Ok(reply)) => Ok(reply),
            Ok(Err(e)) if e.is_command_rejection() => Err(error(e)),
            _ => self.lose_proof(),
        }
    }
    fn lose_proof(&self) -> ! {
        self.control.unknown.store(true, Ordering::Release);
        self.control.terminate();
        std::panic::panic_any(PanicQuiescence::Unknown)
    }
}
impl ReplicaWorker<Command> for Proxy {
    type Output = Reply;
    type Error = Error;
    fn execute(&mut self, request: WorkRequest<Command>) -> Result<Reply> {
        #[cfg(test)]
        let _proof_reuse = boot_cache_tests::RequestValidationGuard::new();
        if request.rank != self.validated.rank.slot {
            return Err(error("numeric proxy slot mismatch"));
        }
        match request.input {
            Command::Stats => Err(error("process owner has no ThreadId; use process_stats")),
            Command::ProcessStats => {
                let NumericReply::Stats(stats) =
                    self.call(request.transaction, &NumericCommand::Stats)?
                else {
                    self.lose_proof()
                };
                if !valid_stats(&self.validated, self.pid, &stats)
                    || stats.host_hits != self.ready.host_hits
                {
                    self.lose_proof();
                }
                Ok(Reply::ProcessStats(stats))
            }
            Command::Compute { context, bucket } => {
                if bucket.owner_rank.get() != self.validated.boot.owner {
                    return Err(error("numeric proxy owner mismatch"));
                }
                let frames = bucket
                    .tokens
                    .iter()
                    .map(NumericToken::encode)
                    .collect::<Vec<_>>();
                self.validated.validate_token_frames(context, &frames)?;
                let command = NumericCommand::Compute {
                    transaction: context.transaction.get(),
                    source: context.source_rank.get(),
                    layer: context.layer,
                    tokens: frames,
                };
                let NumericReply::Results { owner, tokens } =
                    self.call(context.transaction, &command)?
                else {
                    self.lose_proof()
                };
                if owner != self.validated.boot.owner || tokens.len() != bucket.tokens.len() {
                    self.lose_proof();
                }
                let mut results = Vec::with_capacity(tokens.len());
                for (actual, expected) in tokens.into_iter().zip(&bucket.tokens) {
                    if !actual.matches_input(expected, self.validated.boot.hidden) {
                        self.lose_proof();
                    }
                    let t = actual.decode().unwrap_or_else(|_| self.lose_proof());
                    results.push(ExpertResult {
                        transaction: t.transaction,
                        sequence: t.sequence,
                        source_rank: t.source_rank,
                        owner_rank: ParallelRankId::new(owner),
                        source_row: t.source_row,
                        route_slot: t.route_slot,
                        expert: t.expert,
                        output: t.payload,
                    });
                }
                Ok(Reply::Results(results))
            }
            Command::ComputeRoutes { context, routes } => {
                let Some(owner) = routes.first().map(|route| route.metadata.owner_rank) else {
                    return Ok(Reply::RouteResults(Vec::new()));
                };
                if owner != self.validated.rank.owner
                    || routes.iter().any(|route| {
                        route.metadata.owner_rank != owner
                            || route.token.expert.layer != context.layer
                    })
                {
                    return Err(error(
                        "numeric route validation: process owner/layer mismatch",
                    ));
                }
                let frames = routes
                    .iter()
                    .map(|route| NumericRouteToken {
                        route: ExpertRouteFrame::encode(route.metadata),
                        token: NumericToken::encode(&route.token),
                    })
                    .collect::<Vec<_>>();
                self.validated.validate_route_frames(context, &frames)?;
                let command = NumericCommand::ComputeRoutes {
                    transaction: context.transaction.get(),
                    source: context.source_rank.get(),
                    layer: context.layer,
                    tokens: frames,
                };
                let NumericReply::RouteResults { owner, tokens } =
                    self.call(context.transaction, &command)?
                else {
                    self.lose_proof()
                };
                if owner != self.validated.boot.owner || tokens.len() != routes.len() {
                    self.lose_proof();
                }
                let mut results = Vec::with_capacity(tokens.len());
                for (actual, expected) in tokens.into_iter().zip(&routes) {
                    if actual.route.metadata() != expected.metadata
                        || !actual
                            .token
                            .matches_input(&expected.token, self.validated.boot.hidden)
                    {
                        self.lose_proof();
                    }
                    let token = actual.token.decode().unwrap_or_else(|_| self.lose_proof());
                    results.push(ExpertRouteResult {
                        metadata: expected.metadata,
                        result: ExpertResult {
                            transaction: token.transaction,
                            sequence: token.sequence,
                            source_rank: token.source_rank,
                            owner_rank: ParallelRankId::new(owner),
                            source_row: token.source_row,
                            route_slot: token.route_slot,
                            expert: token.expert,
                            output: token.payload,
                        },
                    });
                }
                Ok(Reply::RouteResults(results))
            }
        }
    }
    fn panic_quiescence(&mut self) -> PanicQuiescence {
        self.control.terminate();
        PanicQuiescence::Unknown
    }
    fn shutdown(&mut self) -> Result<()> {
        let result = {
            let mut owner = self.control.owner.lock().unwrap_or_else(|e| e.into_inner());
            owner.as_mut().map(|o| o.shutdown()).transpose()
        };
        match result {
            Ok(Some(_)) => {
                self.control
                    .owner
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take();
                Ok(())
            }
            _ => self.lose_proof(),
        }
    }
}
impl Drop for Proxy {
    fn drop(&mut self) {
        self.control.terminate();
    }
}
fn valid_stats(validated: &ValidatedNumericBoot, pid: u32, s: &NumericProcessStats) -> bool {
    let boot = &validated.boot;
    let payload = validated.payload_bytes;
    s.pid == pid
        && s.owner == boot.owner
        && s.slot == validated.rank.slot.get()
        && s.device == validated.device
        && s.owned_experts.iter().copied().eq(boot
            .expected
            .iter()
            .map(|source| (source.layer, source.expert)))
        && s.host_payload_bytes == payload
        && s.host_read_bytes == payload
        && s.resident_experts == boot.expected.len()
        && s.resident_bytes as u64 == payload
        && s.uploads == boot.expected.len() as u64
        && s.evictions == 0
        && s.pending_upload_bytes == 0
        && !s.quarantined
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericProcessIsolation {
    Standalone,
    Inherited,
}

impl ExpertParallelExecutor {
    pub fn new_numeric_process(
        launch: ProcessLaunch,
        boots: Vec<(ProcessIdentity, NumericBoot)>,
        options: ProcessOwnerConfig,
    ) -> Result<Self> {
        Self::new_numeric_process_scoped(
            launch,
            boots,
            options,
            NumericProcessIsolation::Standalone,
        )
    }
    /// Boots are in membership order; all are metadata-only. The caller must
    /// aggregate root/colocated-device/global-host budgets before invoking this.
    pub fn new_numeric_process_scoped(
        launch: ProcessLaunch,
        boots: Vec<(ProcessIdentity, NumericBoot)>,
        options: ProcessOwnerConfig,
        isolation: NumericProcessIsolation,
    ) -> Result<Self> {
        options.validate().map_err(error)?;
        let validated_boots = boots
            .into_iter()
            .map(|(identity, boot)| {
                ValidatedNumericBoot::build(boot, identity, options.frame_limits)
            })
            .collect::<Result<Vec<_>>>()?;
        let first = validated_boots
            .first()
            .ok_or_else(|| error("empty numeric process group"))?;
        let group = first.group.clone();
        if validated_boots.len() != group.members.len() {
            return Err(error("numeric boot membership mismatch"));
        }
        let timeout = Duration::from_millis(first.boot.placement.timeout_ms);
        let mut instances = BTreeSet::new();
        for (slot, validated) in validated_boots.iter().enumerate() {
            let boot = &validated.boot;
            let identity = validated.identity;
            if validated.group != group
                || identity.rank != group.members[slot]
                || identity.epoch != first.identity.epoch
                || !instances.insert(identity.owner_instance)
                || boot.numeric_profile != first.boot.numeric_profile
                || boot.total_layers != first.boot.total_layers
                || boot.hidden != first.boot.hidden
                || boot.max_rows != first.boot.max_rows
                || boot.checkpoint != first.boot.checkpoint
                || boot.placement != first.boot.placement
            {
                return Err(error("inconsistent scoped numeric boots"));
            }
        }
        let mut proxies = Vec::new();
        let mut controls = Vec::new();
        // Startup is serial: each child keeps only its own payload. On any
        // unwind/Err, the guard explicitly cleans processes before custody leaks.
        let build = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            for validated in validated_boots {
                let identity = validated.identity;
                let boot = &validated.boot;
                let spawned = match isolation {
                    NumericProcessIsolation::Standalone => {
                        Owner::spawn(launch.clone(), identity, boot, options)
                    }
                    NumericProcessIsolation::Inherited => {
                        Owner::spawn_inherited(launch.clone(), identity, boot, options)
                    }
                };
                let owner = match spawned {
                    Ok(owner) => owner,
                    Err(mut e) => {
                        let unknown = e.source.is_quiescence_unknown();
                        if let Some(mut owner) = e.owner.take() {
                            owner.terminate();
                        }
                        if unknown {
                            std::panic::panic_any(PanicQuiescence::Unknown);
                        }
                        return Err(error(e.source));
                    }
                };
                let pid = owner
                    .pid()
                    .ok_or_else(|| error("numeric child missing PID"))?;
                let control = Arc::new(Control {
                    owner: Mutex::new(Some(owner)),
                    stop: AtomicBool::new(false),
                    unknown: AtomicBool::new(false),
                });
                controls.push(Arc::clone(&control));
                let reply = control.owner.lock().unwrap().as_mut().unwrap().execute(
                    ExecutionTransactionId::new(1)?,
                    0,
                    &NumericCommand::Stats,
                );
                let stats = match reply {
                    Ok(NumericReply::Stats(stats)) if valid_stats(&validated, pid, &stats) => stats,
                    _ => {
                        control.unknown.store(true, Ordering::Release);
                        std::panic::panic_any(PanicQuiescence::Unknown);
                    }
                };
                proxies.push(Proxy {
                    control,
                    validated,
                    pid,
                    ready: stats,
                });
            }
            Ok(())
        }));
        let controls = ProcessControls(Arc::new(controls));
        match build {
            Ok(Ok(())) => {}
            other => {
                let mut unknown = other.as_ref().is_err_and(|p| {
                    p.downcast_ref::<PanicQuiescence>() == Some(&PanicQuiescence::Unknown)
                });
                for c in controls.0.iter() {
                    let mut owner = c.owner.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(owner) = owner.as_mut()
                        && let Err(e) = owner.shutdown()
                    {
                        unknown |= e.is_quiescence_unknown();
                    }
                }
                controls.terminate();
                drop(proxies);
                if unknown {
                    std::panic::panic_any(PanicQuiescence::Unknown);
                }
                return match other {
                    Ok(Err(e)) => Err(e),
                    Err(p) => std::panic::resume_unwind(p),
                    _ => unreachable!(),
                };
            }
        }
        let slots = Arc::new(Mutex::new(
            proxies.into_iter().map(Some).collect::<Vec<_>>(),
        ));
        let owned = Arc::clone(&slots);
        let pool = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            DataParallelExecutor::new(
                DataParallelConfig {
                    replicas: group.members.len(),
                    max_outstanding_per_replica: 1,
                    session_capacity: group.members.len(),
                },
                move |slot| {
                    owned.lock().unwrap()[slot.get() as usize]
                        .take()
                        .ok_or_else(|| error("numeric proxy already installed"))
                },
            )
        }));
        let pool = match pool {
            Ok(Ok(pool)) => pool,
            _ => {
                controls.terminate();
                drop(slots);
                std::panic::panic_any(PanicQuiescence::Unknown);
            }
        };
        Ok(Self::from_process_workers(
            ExpertRankWorkers::from_process_pool(group, pool, controls, timeout),
        ))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/parallel/numeric_codec.rs"]
mod codec_tests;

#[cfg(test)]
#[path = "../../tests/unit/parallel/numeric_child_facade.rs"]
mod child_facade_tests;

#[cfg(test)]
#[path = "../../tests/bench/numeric_domain_cpu.rs"]
pub(crate) mod numeric_domain_cpu;

#[cfg(test)]
#[path = "../../tests/unit/parallel/numeric_boot_cache.rs"]
mod boot_cache_tests;

#[cfg(test)]
#[path = "../../tests/unit/parallel/numeric_profile.rs"]
mod numeric_profile_tests;
