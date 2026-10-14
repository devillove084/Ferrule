//! Exact, bounded KV working-set analysis. This module owns no runtime KV pages.
//!
//! [`RecencyModel::PublishRecencySnapshotV1`] measures publish recency, not demand-touch
//! LRU. [`RecencyModel::DeferredAdmissionPrefixLruV1`] simulates each requested capacity
//! independently. Neither model is Ferrule's actual prefix-entry cache or a prediction
//! of another capacity's scheduling, pinning, sharing, or physical memory consumption.
//!
//! All namespaces share one equal-charge logical-page pool. Namespace equality says
//! nothing about equal physical byte sizes. Input decoding and its memory budget belong
//! to the caller; the limits here bound analyzer-owned logical state, not allocator RSS.
//! See `tests/unit/kvset/CONTRACT.md` for the wire contract and integration example.

use serde::{Deserialize, Serialize};
use std::fmt;

mod engine;
pub use engine::{Analyzer, DeferredAnalyzer, analyze};

#[cfg(test)]
#[path = "../../tests/unit/kvset/tests.rs"]
mod tests;

pub const TRACE_SCHEMA: &str = "ferrule-kvset-trace";
pub const TRACE_SCHEMA_VERSION: u32 = 1;
pub const SEMANTICS_VERSION: u32 = 1;

pub type KvSetResult<T> = std::result::Result<T, KvSetError>;

/// Messages never include token contents, request IDs or namespace strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSetError {
    InvalidConfig(&'static str),
    InvalidTrace(&'static str),
    ResourceLimit(&'static str),
    ArithmeticOverflow,
    AllocationFailed,
    Poisoned,
}

impl fmt::Display for KvSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(reason) => write!(f, "invalid KVSET configuration: {reason}"),
            Self::InvalidTrace(reason) => write!(f, "invalid KVSET trace: {reason}"),
            Self::ResourceLimit(resource) => write!(f, "KVSET resource limit: {resource}"),
            Self::ArithmeticOverflow => f.write_str("KVSET checked arithmetic overflow"),
            Self::AllocationFailed => f.write_str("KVSET allocation failed"),
            Self::Poisoned => f.write_str("KVSET analyzer is poisoned"),
        }
    }
}
impl std::error::Error for KvSetError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecencyModel {
    PublishRecencySnapshotV1,
    DeferredAdmissionPrefixLruV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowMode {
    WarmHistory,
    /// Recognized on the wire, but explicitly rejected by this implementation.
    ResetPerWindow,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContentNamespace {
    pub model: String,
    pub adapter: String,
    pub position: String,
    pub layout: String,
    pub backend: String,
    pub plan: String,
    pub run_scope: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamespaceDefinition {
    pub id: String,
    pub namespace: ContentNamespace,
}

/// Logical-state limits; zero is a valid deny-all limit. They do not bound the
/// caller's already decoded event, allocator overhead or process peak RSS.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceLimits {
    pub max_events: usize,
    pub max_publish_positions: usize,
    pub max_unique_keys: usize,
    pub max_identity_token_bytes: usize,
    pub max_pending_requests: usize,
    /// Sum of known prompt/committed tokens retained by all pending requests.
    pub max_pending_tokens: usize,
    /// Includes padding in RawPage.tokens, even though padding is not identity.
    pub max_tokens_per_event: usize,
    pub max_namespaces: usize,
    pub max_namespace_bytes: usize,
    pub max_request_id_bytes: usize,
    pub max_request_history_bytes: usize,
    pub max_histogram_entries: usize,
    pub max_capacity_points: usize,
    pub max_total_capacity_entries: usize,
    /// Total alpha and p parameters before normalization/deduplication.
    pub max_target_parameters: usize,
    /// Total curve/coverage/target rows in one report.
    pub max_report_points: usize,
    /// Bounds temporary per-demand arrays, including deferred hit bitmaps.
    pub max_demand_summary_entries: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_events: 1_000_000,
            max_publish_positions: 4_000_000,
            max_unique_keys: 1_000_000,
            max_identity_token_bytes: 64 * 1024 * 1024,
            max_pending_requests: 4096,
            max_pending_tokens: 4_000_000,
            max_tokens_per_event: 1_000_000,
            max_namespaces: 4096,
            max_namespace_bytes: 1024 * 1024,
            max_request_id_bytes: 1024,
            max_request_history_bytes: 16 * 1024 * 1024,
            max_histogram_entries: 1_000_000,
            max_capacity_points: 65_536,
            max_total_capacity_entries: 1_000_000,
            max_target_parameters: 128,
            max_report_points: 1_000_000,
            max_demand_summary_entries: 4_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceHeader {
    pub schema: String,
    pub schema_version: u32,
    pub semantics_version: u32,
    pub page_size: usize,
    pub recency_model: RecencyModel,
    pub window_mode: WindowMode,
    pub namespaces: Vec<NamespaceDefinition>,
    #[serde(default)]
    pub limits: ResourceLimits,
}

impl TraceHeader {
    pub fn new(page_size: usize, namespaces: Vec<NamespaceDefinition>) -> Self {
        Self {
            schema: TRACE_SCHEMA.into(),
            schema_version: TRACE_SCHEMA_VERSION,
            semantics_version: SEMANTICS_VERSION,
            page_size,
            recency_model: RecencyModel::PublishRecencySnapshotV1,
            window_mode: WindowMode::WarmHistory,
            namespaces,
            limits: ResourceLimits::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceInput {
    pub header: TraceHeader,
    pub events: Vec<TraceEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestId {
    pub id: String,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReuseLimit {
    AllPromptTokens,
    BeforeLastPromptToken,
}

impl ReuseLimit {
    pub fn eligible_tokens(self, prompt_tokens: usize) -> usize {
        match self {
            Self::AllPromptTokens => prompt_tokens,
            Self::BeforeLastPromptToken => prompt_tokens.saturating_sub(1),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DemandEvent {
    pub seq: u64,
    pub request: RequestId,
    pub namespace_id: String,
    pub prompt_tokens: Vec<u32>,
    pub reuse_limit: ReuseLimit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawPage {
    pub tokens: Vec<u32>,
    /// Only tokens[..valid_count] participate in identity. Padding is ignored.
    pub valid_count: usize,
}

/// A producer assertion of an irrevocably committed FULL prefix, not a delta.
/// Actual committed decode tokens may extend beyond the original prompt. Neither
/// proposed tokens nor a successful Outcome establish commitment. This analyzer
/// verifies consistency, not backend completion; exporters must prove that boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishEvent {
    pub seq: u64,
    pub request: RequestId,
    pub namespace_id: String,
    pub pages: Vec<RawPage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestStatus {
    Success,
    Failed,
    Cancelled,
    Abandoned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActualObservation {
    pub reused_prefix_tokens: usize,
    pub reused_pages: usize,
    pub cache_capacity_pages: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeEvent {
    pub seq: u64,
    pub request: RequestId,
    pub status: RequestStatus,
    pub actual: Option<ActualObservation>,
}

/// Internally tagged, flat JSON: {"seq":0,"type":"demand",...}.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum TraceEvent {
    Demand(DemandEvent),
    Publish(PublishEvent),
    Outcome(OutcomeEvent),
}

impl TraceEvent {
    pub fn seq(&self) -> u64 {
        match self {
            Self::Demand(e) => e.seq,
            Self::Publish(e) => e.seq,
            Self::Outcome(e) => e.seq,
        }
    }

    pub fn request(&self) -> &RequestId {
        match self {
            Self::Demand(e) => &e.request,
            Self::Publish(e) => &e.request,
            Self::Outcome(e) => &e.request,
        }
    }
}

/// Exact fraction. Configuration fractions must satisfy 0 <= numerator <= denominator.
/// Report ratios are unsimplified counts; an undefined ratio is represented by None.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ratio {
    pub numerator: u64,
    pub denominator: u64,
}

impl Ratio {
    pub fn new(numerator: u64, denominator: u64) -> KvSetResult<Self> {
        let ratio = Self {
            numerator,
            denominator,
        };
        ratio.validate()?;
        Ok(ratio)
    }

    pub fn validate(self) -> KvSetResult<()> {
        if self.denominator == 0 || self.numerator > self.denominator {
            return Err(KvSetError::InvalidConfig("ratio must be in [0,1]"));
        }
        Ok(())
    }

    pub fn ceil_mul(self, n: u64) -> KvSetResult<u64> {
        self.validate()?;
        let value =
            (u128::from(n) * u128::from(self.numerator)).div_ceil(u128::from(self.denominator));
        u64::try_from(value).map_err(|_| KvSetError::ArithmeticOverflow)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetBasis {
    TheoreticalReusable,
    EligibleTokens,
    FullPrompt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageDenominator {
    Opportunity,
    AllDemands,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementWindow {
    /// Demand membership is [start_seq, end_seq); all events still update history.
    pub start_seq: u64,
    pub end_seq: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnalyzerConfig {
    /// None: all snapshot-model breakpoints. Deferred requires a nonempty list.
    pub capacities: Option<Vec<usize>>,
    pub alphas: Vec<Ratio>,
    pub p_values: Vec<Ratio>,
    pub target_basis: TargetBasis,
    pub coverage_denominator: CoverageDenominator,
    pub measurement_window: Option<MeasurementWindow>,
    /// Separate planning constraint, not a truncation of modeled history.
    /// In deferred mode this must also be an explicitly sampled capacity.
    pub capacity_budget: Option<usize>,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            capacities: None,
            alphas: vec![Ratio {
                numerator: 1,
                denominator: 1,
            }],
            p_values: vec![
                Ratio {
                    numerator: 95,
                    denominator: 100,
                },
                Ratio {
                    numerator: 99,
                    denominator: 100,
                },
            ],
            target_basis: TargetBasis::TheoreticalReusable,
            coverage_denominator: CoverageDenominator::Opportunity,
            measurement_window: None,
            capacity_budget: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "distance", rename_all = "snake_case")]
pub enum StackDistance {
    Cold,
    Warm(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DemandCapacity {
    pub capacity_pages: usize,
    /// All page probes use the same demand-entry snapshot, including after a hole.
    pub page_hits: Vec<bool>,
    pub contiguous_prefix_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DemandSummary {
    pub seq: u64,
    pub request: RequestId,
    pub measured: bool,
    pub prompt_tokens: usize,
    pub eligible_tokens: usize,
    pub theoretical_reusable_tokens: usize,
    pub first_cold_page: Option<usize>,
    /// Some only for PublishRecencySnapshotV1; never predicts deferred hits.
    pub page_distances: Option<Vec<StackDistance>>,
    /// Populated only for DeferredAdmissionPrefixLruV1.
    pub capacity_results: Vec<DemandCapacity>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadCounts {
    pub demands: u64,
    pub prompt_tokens: u64,
    pub eligible_tokens: u64,
    pub theoretical_reusable_tokens: u64,
    pub page_references: u64,
    pub cold_page_references: u64,
    pub requests_with_cold_pages: u64,
    pub no_eligible_tokens: u64,
    pub cold_prefix_requests: u64,
    pub opportunity_requests: u64,
    pub no_opportunity_requests: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeCounts {
    pub success: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub abandoned: u64,
    pub abandoned_at_trace_end: u64,
    pub pending: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActualTotals {
    pub observed_requests: u64,
    pub reused_prefix_tokens: u64,
    pub reused_pages: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityPoint {
    pub capacity_pages: usize,
    pub page_hits: u64,
    pub page_reference_hit_rate: Option<Ratio>,
    pub contiguous_prefix_tokens: u64,
    pub contiguous_prefix_token_hit_rate_theoretical: Option<Ratio>,
    pub contiguous_prefix_token_hit_rate_eligible: Option<Ratio>,
    pub contiguous_prefix_token_hit_rate_prompt: Option<Ratio>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoveragePoint {
    pub capacity_pages: usize,
    pub achieved_requests: u64,
    pub coverage_opportunity: Option<Ratio>,
    pub coverage_all_demands: Option<Ratio>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestTargetCoverage {
    pub alpha: Ratio,
    /// Mathematical target > theoretical reusable prefix, independent of capacity.
    pub unreachable_target_requests: u64,
    /// Opportunity-bearing, mathematically reachable requests not met at the budget.
    pub beyond_budget_requests: Option<u64>,
    pub points: Vec<CoveragePoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum CapacityRequirement {
    Exact { capacity: usize },
    AtSampledCapacity { capacity: usize },
    BeyondBudget { budget: usize },
    Unreachable,
    NotReachedAtSampledCapacities,
    Undefined,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetCapacity {
    pub alpha: Ratio,
    pub p: Ratio,
    pub denominator: CoverageDenominator,
    pub requirement: CapacityRequirement,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyzerUsage {
    pub events: usize,
    pub publish_positions: usize,
    pub unique_keys: usize,
    pub identity_token_bytes: usize,
    pub pending_requests: usize,
    pub pending_tokens: usize,
    pub request_history_bytes: usize,
    pub histogram_entries: usize,
    pub finite_cache_entries: usize,
    /// Retained identity + pending token payload only, not peak heap or RSS.
    pub retained_token_payload_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisReport {
    pub schema_version: u32,
    pub semantics_version: u32,
    pub recency_model: RecencyModel,
    pub page_size: usize,
    pub finalized: bool,
    pub config: AnalyzerConfig,
    pub workload: WorkloadCounts,
    pub outcomes: OutcomeCounts,
    pub actual: ActualTotals,
    pub theoretical_opportunity_fraction_of_prompt: Option<Ratio>,
    pub capacity_points: Vec<CapacityPoint>,
    pub request_target_coverage: Vec<RequestTargetCoverage>,
    pub target_capacities: Vec<TargetCapacity>,
    pub usage: AnalyzerUsage,
}
