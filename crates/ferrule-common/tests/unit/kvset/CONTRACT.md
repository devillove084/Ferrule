# Restored KVSET v1 contract

This is the implementation contract for downstream runtime/CLI integration, not a
claim that the lost source was recovered byte-for-byte. Authoritative Rust DTOs
are in `mod.rs`. Historical implementation evidence: `docs/temp_ferrule_impl.md`
86426–86615, 86981–87097, 90152–90320. The earlier proposals in
`docs/temp_kv_set.md` are not alternative runtime contracts.

## API

```rust
use ferrule_common::kvset::*;

fn analyze(input: &TraceInput, config: AnalyzerConfig)
    -> KvSetResult<AnalysisReport>;

// Same signatures on DeferredAnalyzer, but it requires its matching header model.
fn Analyzer::new(header: TraceHeader, config: AnalyzerConfig)
    -> KvSetResult<Analyzer>;
fn Analyzer::ingest(&mut self, event: &TraceEvent)
    -> KvSetResult<Option<DemandSummary>>;
fn Analyzer::report(&mut self) -> KvSetResult<AnalysisReport>;
fn Analyzer::finish(self) -> KvSetResult<AnalysisReport>;
```

`KvSetResult<T>` uses the module-local `KvSetError`, not common's general error.
Errors carry static reasons only, never tokens or caller identifiers. `ingest`
and `report` poison the instance on any error; subsequent ingest/report/finish
return `Poisoned`. Partial internal mutation before an error is not rolled back
and is never reportable. `report` does not close pending requests. `finish`
consumes the analyzer and classifies measured pending requests as
`abandoned_at_trace_end`, clears pending token payloads, and reports finalized.
No elapsed time or RSS is measured in common.

## Wire DTOs

All public DTOs derive serde Serialize/Deserialize. Structs reject unknown and
duplicate fields. Input decoding and byte ceilings remain the caller's job.

```text
TraceInput { header: TraceHeader, events: Vec<TraceEvent> }
TraceHeader {
  schema: String = "ferrule-kvset-trace",
  schema_version: u32 = 1,
  semantics_version: u32 = 1,
  page_size: usize > 0,
  recency_model: RecencyModel,
  window_mode: WindowMode,
  namespaces: Vec<NamespaceDefinition>,
  limits: ResourceLimits // omitted => documented finite defaults
}
NamespaceDefinition { id: String, namespace: ContentNamespace }
ContentNamespace { model, adapter, position, layout, backend, plan, run_scope: String }
RequestId { id: String, generation: u64 }
RawPage { tokens: Vec<u32>, valid_count: usize }
```

Namespace fields must be nonempty; use `"none"` for no adapter. IDs are trace-local
lookup names; equal complete descriptors under two IDs normalize to the same
namespace. Different descriptors isolate history. One capacity is shared across
ALL namespaces in equal-charge logical pages, not bytes.

Rust event variants are **newtype variants**:

```rust
TraceEvent::Demand(DemandEvent {
    seq, request, namespace_id, prompt_tokens, reuse_limit,
})
TraceEvent::Publish(PublishEvent {
    seq, request, namespace_id, pages,
})
TraceEvent::Outcome(OutcomeEvent {
    seq, request, status, actual,
})
```

JSON is flat, internally tagged by `type`, snake_case:

```json
{"seq":0,"type":"demand","request":{"id":"r","generation":0},"namespace_id":"n","prompt_tokens":[1,2],"reuse_limit":"all_prompt_tokens"}
```

`seq: u64` strictly increases; gaps are legal (unlike runtime trace watermarks).
Request textual IDs cannot overlap while pending. After Outcome, reuse requires
a strictly larger generation. Namespace on Publish must normalize to the Demand
namespace. Unknown request, wrong incarnation, post-terminal publication, or
repeated terminal is invalid.

`ReuseLimit::{AllPromptTokens, BeforeLastPromptToken}` defines eligible tokens
`E=L` or `L.saturating_sub(1)`. `RequestStatus` is Success/Failed/Cancelled/Abandoned.
`actual: Option<ActualObservation>` is null/absent or
`{reused_prefix_tokens, reused_pages, cache_capacity_pages: usize}`. It records
observed admission reuse independently of modeled hits. Tokens cannot exceed the
Demand eligibility; page count equals ceil(tokens/page_size) and cannot exceed
observed page capacity. Outcome success does not imply Publish.

## Publication and identity

Publish is a producer assertion of a **full irrevocably committed prefix**, never
a delta. It may be empty, repeated, or extend into actually committed decoded
tokens beyond Demand. It cannot shorten an earlier committed prefix or change any
known prompt/committed token. The analyzer checks consistency, not backend
completion; the runtime projector must prove the commit boundary.

Pages are canonical: all non-tail valid counts equal page_size; a tail has
`1 <= valid_count <= tokens.len() <= page_size`. Tail padding is ignored for
identity but charged to the event token limit. Interner equality uses normalized
namespace + parent intern ID + valid token slice (its length is the valid count).
Full equality resolves hash collisions. Partial/full versions are different;
full pages never implicitly publish short aliases. Internal IDs/hashes are not
portable content identities; physical KvPageId never participates.

## Models and measurements

- `PublishRecencySnapshotV1`: Demand is one read-only snapshot, no touch or insert.
  Publish refreshes/inserts canonical pages in ascending offset order. Fenwick
  distance is `sum(end)-sum(last)`; hit iff `distance < capacity`. The histogram
  yields exact breakpoints without replaying each capacity.
- `DeferredAdmissionPrefixLruV1`: each explicit capacity has its own finite LRU.
  Demand snapshots all probes first, then touches only the contiguous hit prefix
  in offset order. Misses and hits beyond a hole do not touch or insert. Publish
  inserts/refreshes as above. Snapshot Fenwick distances are NOT its hit predictor.
  Simulation uses linear resident lists, suitable for bounded comparison workloads.

`DemandSummary.page_distances` is Some only for the snapshot model;
`capacity_results` is populated only for deferred. Summaries contain all page
probes even after the first miss, but only contiguous tokens count as reuse.

T is the continuous reusable prefix in infinite **already-published** history at
Demand time. Default alpha target is ceil(alpha*T). Other bases are EligibleTokens
and FullPrompt. T=0 never counts as achieved, including alpha=0. Page references,
contiguous tokens, and request coverage are separate metrics. Token ratios use
T/E/L denominators; request coverage uses opportunity/all-demand denominators.
Zero denominators are None => JSON null, not zero or one.

`AnalyzerConfig` fields:

```text
capacities: Option<Vec<usize>>          // None => all snapshot breakpoints
alphas: Vec<Ratio>                     // default [1/1]
p_values: Vec<Ratio>                   // default [95/100,99/100]
target_basis: TargetBasis              // default TheoreticalReusable
coverage_denominator: CoverageDenominator // default Opportunity
measurement_window: Option<MeasurementWindow {start_seq,end_seq: u64}>
capacity_budget: Option<usize>         // planning constraint, not eviction
Ratio { numerator: u64, denominator: u64 }
```

Config uses defaults for omitted fields. Ratios must be in [0,1], denominator>0;
normalization and ordering use integer arithmetic. Lists are deduplicated after
limits are checked. Explicit capacities must be nonempty. Deferred requires them,
and its budget, if supplied, must be sampled. Snapshot exact target requirements
remain exact even if output capacities are sparse. Window is [start,end), including
an empty window; all events still maintain history. Demand membership controls
terminal/actual accounting even if Outcome arrives after the window.
`ResetPerWindow` is recognized but rejected.

CapacityRequirement is tagged `status`, with capacity/budget where applicable:
Exact, AtSampledCapacity, BeyondBudget, Unreachable,
NotReachedAtSampledCapacities, Undefined. An empty denominator is Undefined;
otherwise p=0 is always Exact(0), including deferred capacities=[2]. Mathematical
unreachability is determined from the entire cohort, not lack of a large enough
sample. Deferred only reports a sampled satisfying capacity; a configured sampled
budget can prove BeyondBudget at that budget. `unreachable_target_requests` counts
mathematical target>T independently of no-opportunity. `beyond_budget_requests`
counts opportunity-bearing mathematically reachable requests failing at budget.

## Bounds and privacy

ResourceLimits in mod.rs bounds events, positions, identity keys/tokens, pending
requests/tokens, namespaces/bytes, request history bytes, histogram entries,
capacity points/total resident entries, target parameters, report points and
per-demand summary arrays. Zero denies that resource. Publish preflights all
page shapes/input tokens/positions and pending growth before copying or interning.
Histogram/report breakpoints check limits before insertion. Exact history is never
silently evicted. Allocator capacity overhead is not payload accounting or RSS.
Caller-owned decoded inputs and accumulated returned reports are outside these
limits; untrusted trace-provided limits must be capped by the importing caller.

This analyzer has no KV/page ownership, runtime hooks, file I/O, CUDA, placement or
automatic eviction control. Trace token IDs can disclose sensitive inputs/outputs.
Actual prefix-entry LRU must remain a separately validated runtime report, never
renamed as these ideal page curves.

## Fixtures and reproducible checks

- `tests/fixtures/kvset/kvset_trace_v1.json` and
  `tests/fixtures/kvset/kvset_config_v1.json` describe two measured three-token
  prefixes. At C=0,1,2,3: page hits=0,1,3,4; tokens=0,0,5,6.
- `cargo test --locked -p ferrule-common --lib kvset` runs production-state tests,
  independent VecDeque/full-prefix-key oracles and strict serde Value tests.
- `tests/unit/kvset/json_contract_probe.rs` is intentionally outside the common
  module graph so common needs no serde_json dependency. It can be compiled with `rustc --test`
  against freshly built common and serde_json artifacts (use dependency paths
  printed by `cargo build --locked -p ferrule-common -v`). It decodes both real
  fixtures, validates JSON/JSONL round trips and report null/strict-error behavior.
  CLI integration tests may include this source as a test module because CLI
  already depends on serde_json.

These checks are CPU contract tests, not measured production workload or GPU
performance evidence.
