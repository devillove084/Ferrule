//! Opt-in bounded KV observations. Token IDs are sensitive; no hook performs I/O.
//! Physical page IDs are diagnostics, never logical content identities.
use ferrule_common::kvset::ContentNamespace;
use serde::{Deserialize, Serialize};

mod kvset_adapter;
mod replay;
#[cfg(test)]
#[path = "../../tests/unit/engine/kv_trace/mod.rs"]
mod tests;
mod validation;
pub use kvset_adapter::{project_runtime_kv_trace, project_runtime_kv_trace_with_limits};
pub use replay::replay_runtime_prefix_lru;

pub const RUNTIME_KV_TRACE_SCHEMA: &str = "ferrule-runtime-kv-trace";
pub const RUNTIME_KV_TRACE_VERSION: u32 = 2;
pub type TraceResult<T> = Result<T, RuntimeKvTraceError>;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeKvTraceError(pub &'static str);
impl std::fmt::Display for RuntimeKvTraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for RuntimeKvTraceError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeKvTraceConfig {
    pub run_scope: String,
    pub stable_content_namespace: Option<ContentNamespace>,
    pub max_events: usize,
    pub max_token_bytes: usize,
    pub max_observations: usize,
}
impl Default for RuntimeKvTraceConfig {
    fn default() -> Self {
        Self {
            run_scope: String::new(),
            stable_content_namespace: None,
            max_events: 65_536,
            max_token_bytes: 4 * 1024 * 1024,
            max_observations: 4096,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeKvTraceHeader {
    pub schema: String,
    pub schema_version: u32,
    pub config: RuntimeKvTraceConfig,
    pub page_size: usize,
    pub prefix_capacity_pages: usize,
    /// Exact runtime model/backend/device/plan/owner tuple, scoped by run_scope.
    pub namespace: [u64; 5],
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeKvTraceStatus {
    pub overflow: bool,
    pub unsupported: bool,
    pub observation_missing: bool,
}
impl RuntimeKvTraceStatus {
    pub fn is_complete(self) -> bool {
        !self.overflow && !self.unsupported && !self.observation_missing
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeKvTraceUsage {
    pub events: u64,
    pub token_bytes: u64,
    pub observations: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeKvTraceSnapshot {
    pub header: RuntimeKvTraceHeader,
    pub start_seq: u64,
    pub next_seq: u64,
    pub lifecycle_events: Vec<RuntimeKvTraceEvent>,
    pub tokens: Vec<u32>,
    pub status: RuntimeKvTraceStatus,
    pub complete: bool,
    pub finalized: bool,
    pub usage: RuntimeKvTraceUsage,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenSpan {
    pub start: usize,
    pub len: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LookupOutcome {
    Hit,
    Miss,
    Bypass,
    Error,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemovalReason {
    Capacity,
    PagePressure,
    Drain,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalStatus {
    Success,
    Failed,
    Cancelled,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeKvTraceEvent {
    pub seq: u64,
    pub kind: RuntimeKvTraceKind,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum RuntimeKvTraceKind {
    Demand {
        observation: u64,
        request: u64,
        session: u64,
        tokens: TokenSpan,
    },
    Lookup {
        observation: Option<u64>,
        outcome: LookupOutcome,
        entry: Option<u64>,
        matched_tokens: usize,
        generation: u64,
    },
    UnpinAck {
        entry: u64,
    },
    AdmissionPublished {
        observation: u64,
        actual_tokens: usize,
        actual_pages: usize,
    },
    /// `reservation` is the execution transaction serial, scoped by observation;
    /// ReserveSucceeded separately records the page-manager reservation identity.
    ExecutionInput {
        observation: u64,
        reservation: u64,
        start: usize,
        tokens: TokenSpan,
    },
    LogicalKvPublished {
        observation: u64,
        reservation: u64,
        committed_tokens: usize,
    },
    ReservationAborted {
        observation: u64,
        reservation: u64,
    },
    ReserveSucceeded {
        observation: Option<u64>,
        reservation: u64,
        allocated_pages: usize,
        cow: bool,
    },
    CowReplacement {
        reservation: u64,
        logical_page: usize,
        source: u32,
        replacement: u32,
    },
    SnapshotCaptured {
        snapshot: u64,
        committed_tokens: usize,
        pages: usize,
    },
    SnapshotForkPublished {
        snapshot: u64,
        observation: u64,
    },
    SnapshotReleased {
        snapshot: u64,
        retiring_pages: usize,
    },
    PrefixEntryPublished {
        entry: u64,
        tokens: TokenSpan,
        charge_pages: usize,
        generation: u64,
        replaced: bool,
    },
    /// The existing radix removal API exposes exact key and generation, not ID.
    /// Replay resolves this against its entry index; no online mirror is kept.
    PrefixEntryRemoved {
        tokens: TokenSpan,
        charge_pages: usize,
        generation: u64,
        reason: RemovalReason,
    },
    BackendReleaseAck {
        pages: usize,
    },
    RetirementConfirmed {
        pages: usize,
    },
    Cancellation {
        observation: u64,
    },
    Terminal {
        observation: u64,
        status: TerminalStatus,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Observation {
    pub id: u64,
    pub request: u64,
    pub session: u64,
    pub execution: Option<u64>,
}
/// A single owner-local recorder. All hot-path storage is reserved at enable.
/// Observation records correlate identities only; they grant no KV authority.
pub(crate) struct RuntimeKvRecorder {
    header: RuntimeKvTraceHeader,
    events: Vec<RuntimeKvTraceEvent>,
    tokens: Vec<u32>,
    observations: Vec<Observation>,
    status: RuntimeKvTraceStatus,
    usage: RuntimeKvTraceUsage,
    start_seq: u64,
    next_seq: u64,
    next_observation: u64,
    finalized: bool,
}
impl RuntimeKvRecorder {
    pub(crate) fn new(
        config: RuntimeKvTraceConfig,
        page_size: usize,
        capacity: usize,
        namespace: [u64; 5],
    ) -> TraceResult<Self> {
        if config.run_scope.trim().is_empty() || config.run_scope.len() > 4096 || page_size == 0 {
            return Err(RuntimeKvTraceError(
                "trace requires a bounded nonempty run scope and page size",
            ));
        }
        if let Some(n) = &config.stable_content_namespace {
            for s in [
                &n.model,
                &n.adapter,
                &n.position,
                &n.layout,
                &n.backend,
                &n.plan,
                &n.run_scope,
            ] {
                if s.trim().is_empty() || s.len() > 4096 {
                    return Err(RuntimeKvTraceError("invalid content namespace"));
                }
            }
        }
        let mut events = Vec::new();
        let mut tokens = Vec::new();
        let mut observations = Vec::new();
        events
            .try_reserve_exact(config.max_events)
            .map_err(|_| RuntimeKvTraceError("event allocation failed"))?;
        tokens
            .try_reserve_exact(config.max_token_bytes / 4)
            .map_err(|_| RuntimeKvTraceError("token allocation failed"))?;
        observations
            .try_reserve_exact(config.max_observations)
            .map_err(|_| RuntimeKvTraceError("observation allocation failed"))?;
        Ok(Self {
            header: RuntimeKvTraceHeader {
                schema: RUNTIME_KV_TRACE_SCHEMA.into(),
                schema_version: 2,
                config,
                page_size,
                prefix_capacity_pages: capacity,
                namespace,
            },
            events,
            tokens,
            observations,
            status: Default::default(),
            usage: Default::default(),
            start_seq: 0,
            next_seq: 0,
            next_observation: 0,
            finalized: false,
        })
    }
    pub(crate) fn unsupported(&mut self) {
        self.status.unsupported = true;
    }
    pub(crate) fn observation(&mut self, session: u64) -> Option<u64> {
        let result = self
            .observations
            .iter()
            .find(|o| o.session == session)
            .map(|o| o.id);
        if result.is_none() {
            self.status.observation_missing = true;
        }
        result
    }
    pub(crate) fn record(&mut self, kind: RuntimeKvTraceKind) {
        self.record_tokens(&[], |_| kind);
    }
    pub(crate) fn record_tokens(
        &mut self,
        tokens: &[u32],
        make: impl FnOnce(TokenSpan) -> RuntimeKvTraceKind,
    ) {
        if self.finalized || self.status.overflow {
            return;
        }
        let Some(end) = self.tokens.len().checked_add(tokens.len()) else {
            self.status.overflow = true;
            return;
        };
        let Some(next) = self.next_seq.checked_add(1) else {
            self.status.overflow = true;
            return;
        };
        let bytes = (tokens.len() as u64)
            .checked_mul(4)
            .and_then(|n| self.usage.token_bytes.checked_add(n));
        if self.events.len() >= self.header.config.max_events
            || end > self.header.config.max_token_bytes / 4
            || bytes.is_none()
        {
            self.status.overflow = true;
            return;
        }
        let span = TokenSpan {
            start: self.tokens.len(),
            len: tokens.len(),
        };
        self.tokens.extend_from_slice(tokens);
        self.events.push(RuntimeKvTraceEvent {
            seq: self.next_seq,
            kind: make(span),
        });
        self.next_seq = next;
        self.usage.events = next;
        self.usage.token_bytes = bytes.unwrap();
    }
    pub(crate) fn demand(&mut self, request: u64, session: u64, tokens: &[u32]) {
        if self.finalized || self.status.overflow {
            return;
        }
        if self.observations.len() >= self.header.config.max_observations
            || self.next_observation == u64::MAX
        {
            self.status.overflow = true;
            return;
        }
        if self
            .observations
            .iter()
            .any(|o| o.session == session || o.request == request)
        {
            self.status.observation_missing = true;
            return;
        }
        let id = self.next_observation;
        self.next_observation += 1;
        self.usage.observations = self.next_observation;
        self.observations.push(Observation {
            id,
            request,
            session,
            execution: None,
        });
        self.record_tokens(tokens, |tokens| RuntimeKvTraceKind::Demand {
            observation: id,
            request,
            session,
            tokens,
        });
    }
    pub(crate) fn input(&mut self, session: u64, reservation: u64, start: usize, tokens: &[u32]) {
        let Some(observation) = self.observation(session) else {
            return;
        };
        let o = self
            .observations
            .iter_mut()
            .find(|o| o.id == observation)
            .unwrap();
        if o.execution.replace(reservation).is_some() {
            self.status.observation_missing = true;
        }
        self.record_tokens(tokens, |tokens| RuntimeKvTraceKind::ExecutionInput {
            observation,
            reservation,
            start,
            tokens,
        });
    }
    pub(crate) fn finish_execution(
        &mut self,
        session: u64,
        reservation: u64,
        committed: Option<usize>,
    ) {
        let Some(o) = self.observations.iter_mut().find(|o| o.session == session) else {
            self.status.observation_missing = true;
            return;
        };
        if o.execution != Some(reservation) {
            return;
        } // cleanup retries must not duplicate observations
        o.execution = None;
        let observation = o.id;
        self.record(match committed {
            Some(committed_tokens) => RuntimeKvTraceKind::LogicalKvPublished {
                observation,
                reservation,
                committed_tokens,
            },
            None => RuntimeKvTraceKind::ReservationAborted {
                observation,
                reservation,
            },
        });
    }
    pub(crate) fn cancellation(&mut self, request: u64) {
        if let Some(o) = self.observations.iter().find(|o| o.request == request) {
            self.record(RuntimeKvTraceKind::Cancellation { observation: o.id });
        }
    }
    pub(crate) fn terminal(&mut self, request: u64, status: TerminalStatus) {
        let Some(i) = self.observations.iter().position(|o| o.request == request) else {
            self.status.observation_missing = true;
            return;
        };
        let o = self.observations.swap_remove(i);
        if o.execution.is_some() {
            self.status.observation_missing = true;
        }
        self.record(RuntimeKvTraceKind::Terminal {
            observation: o.id,
            status,
        });
    }
    pub(crate) fn drain(&mut self, finalized: bool) -> Option<RuntimeKvTraceSnapshot> {
        if self.finalized {
            return None;
        }
        self.finalized = finalized;
        let snapshot = RuntimeKvTraceSnapshot {
            header: self.header.clone(),
            start_seq: self.start_seq,
            next_seq: self.next_seq,
            lifecycle_events: self.events.clone(),
            tokens: self.tokens.clone(),
            status: self.status,
            complete: finalized && self.status.is_complete() && self.observations.is_empty(),
            finalized,
            usage: self.usage,
        };
        self.events.clear();
        self.tokens.clear();
        self.start_seq = self.next_seq;
        Some(snapshot)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeKvProjectionLimits {
    pub max_events: usize,
    pub max_tokens: usize,
    pub max_publish_pages: usize,
}
impl Default for RuntimeKvProjectionLimits {
    fn default() -> Self {
        Self {
            max_events: 1_000_000,
            max_tokens: 16_000_000,
            max_publish_pages: 4_000_000,
        }
    }
}

/// Exact replay of this observed capacity, not a counterfactual for other capacities.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimePrefixReplay {
    pub lookup_hits: u64,
    pub lookup_misses: u64,
    pub published_admissions: u64,
    pub actual_reused_tokens: u64,
    pub resident_entries: usize,
    pub charged_pages: usize,
    pub removed_charge_pages: usize,
    pub backend_released_pages: usize,
    pub confirmed_pages: usize,
}
