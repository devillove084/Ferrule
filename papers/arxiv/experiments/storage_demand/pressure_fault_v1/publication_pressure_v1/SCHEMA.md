# Typed public record schema

`records.jsonl.gz` is UTF-8 JSON Lines compressed with gzip (no filename,
mtime=0). Every decompressed line has:

```json
{
  "schema": "ferrule.pressure-public-record.v1",
  "kind": "server_event",
  "point": "small",
  "source": {
    "path": "small-blocking/server.log",
    "sha256": "64-lowercase-hex-characters",
    "line": 16
  },
  "data": {}
}
```

The example's digest is a type description, not an evidence digest. Actual
records contain the original source-file SHA256 from `source-manifest.json`.
Line numbers are one-based and present for line-oriented input. `point` is
`small`, `middle`, `big`, or null for study/handoff records. Paths are relative
to `artifacts-go-v2-20261015-02`. A projection's primary source pointer does
not assert that its serialization is byte-identical to that source; composite
records also use the manifest's registration, cases, assessment, attempt and
scalar receipts. The extraction is explicit in `curate.py`.

## Record types

| Kind | Count | Data and scope |
|---|---:|---|
| `study` | 1 | Final denominator, registration schedule/limits, preserved candidate and source-role identities; prespecified geometry/fixed options from the separately hashed plan |
| `configuration` | 3 | Requested two-knob settings, fixed read/scheduler options, prior header-preflight scope |
| `source_options` | 3 | Actual startup metadata options; physical admission fields remain null |
| `host_staging_report` | 3 | Actual blocking host-options line |
| `server_event` | 237 | 90 invocation summaries, 78 token enqueues, 60 request lifecycle gates, 3 shutdown summaries and 6 startup reports |
| `lifecycle` | 3 | Point time origins/endpoints, readiness, request denominator, NFS phases and non-release closure evidence |
| `custody` | 3 | Supervisor outcome, authenticated subreaper drain/ECHILD and reaped-entry projections, not fresh OS inspection |
| `coordinator-pre-idle`, `worker-pre-idle`, `coordinator-post-idle` | 3 each | Recorded all-eight idle windows, consecutive accepted indices and typed sample values |
| `locks-released`, `locks-reprobed` | 3 each | Original local lock-release/reprobe observations |
| `library_validation` | 6 | Ready/after-request pathname-pin result, library basenames/hashes, ignored non-library mapping and live-memory-hash limitation |
| `request` | 12 | Frozen prompt/reference, complete raw HTTP/SSE evidence, actual worker trace join, quality flags and NFS window |
| `memory_sample` | 108 | Recorded monotonic sample interval, selected-device memory/utilization, tree RSS/PSS and coverage/attribution status |
| `phase_timing` | 1,074 | Every named helper start/end wall/thread-CPU observation; 537 paired completed spans |
| `nfs_snapshot` | 42 | Shared checkpoint-client mount counters before/after requests and lifecycle phases, timestamps and parse errors |
| `scalar_reference` | 3 | Selected sealed `point-scalars.json` values, used only as a cross-check of independently derived raw values |
| `handoff`, `handoff_idle`, `handoff_locks` | 1 each | Retained final custody/idle/lock handoff, without any new query |

Total: **1,516 records**. The run has **30 decoder invocations per point**,
not 30 requests. The additional decode/terminal work is not assigned an
invented HTTP token boundary.

## Transformations and omissions

- All invocation fields are retained. The JSON-encoded strings
  `expert_read_pipeline`, `expert_read_pipeline_delta`, and `summary_json`
  become typed JSON objects. Rust `Some(n)`, `Some(true/false)`, and `None`
  in `generated_tokens`/`tokens_reconcile` become number/bool/null.
- The original HTTP SSE `data` strings and their timestamps are retained.
  They are parsed by the checker, not reconstructed from final output text.
  `actual.output_token_ids` remains null; actual IDs come from worker enqueue
  events bound to the actual observed completion ID. Content events are not
  asserted to be token boundaries.
- Duplicate invocation arrays nested in each observed request are omitted;
  the complete server-event sequence is retained once. Huge repeated
  lifecycle/assessment/report trees are not copied wholesale.
- Public projections omit operator home paths, endpoints/listening URLs,
  model-mount coordinates, GPU UUIDs, PID/birth identities, lock filenames,
  launch envelopes, full environment, and unrelated process diagnostics.
  Library basenames/hashes, source-relative Rust role paths, actual request
  IDs/token IDs, monotonic observations, semantic states, and counters remain.
  These omissions narrow the public custody evidence; hashes point back to
  the original private receipts for deeper audits.
- NFS snapshots retain only the relevant shared NFS client's typed counters,
  not unrelated mount tables or backend hostnames. Idle GPU strings become
  device-index/memory/utilization rows. These are projections of retained
  observations, never fresh measurements.
- `history.json` retains the original STOP values and the separately hashed
  historical root analysis. It is never an input to `derive()` or the figure.
  The root analysis's historical invocation totals cannot be pooled into the
  successful three-point denominator.

## Aggregation rules

1. Sum only **invocation deltas**: logical checkpoint reader bytes/calls/source
   checks, metadata calls, expert hits/uploads/evictions, artifact uploads and
   bytes, H2D copies and bytes. The checker requires integer, nonmissing,
   nonreset counters and exactly 30 ordered successful invocations per point.
2. Owner cache gauges, owner read lifetime statistics and owner high-water
   marks are **not sums**. Use the final closure snapshot; reconcile lifetime
   read counters against the corresponding deltas. Cache peak is the maximum
   observed owner peak, not the sum of repeated lifetime peaks.
3. Cache charged bytes = resident + pending upload + scratch. Unknown
   quarantine is a subset. At closure, ending entries = installs − evictions.
   These are logical quantities, not allocator/RSS decomposition.
4. Memory maximum = maximum over the 36 retained complete samples for the
   relevant component. It remains a sampled lower bound on its lifetime peak,
   not a continuous-time upper bound. Missing data must not become zero.
5. Recompute shared-client NFS deltas by subtracting the paired phase
   snapshots; do not substitute them for logical checkpoint reads or backend
   disk traffic. Keep observer windows distinct from the HTTP transport span.
6. Match actual SSE completion ID to worker ID, preserve whole-log admission
   and terminal order, reconcile usage/DONE/text, then verify the three exact
   eight-ID references. A complete worker trace cannot repair incomplete
   HTTP evidence. Hello2 has no exact-ID quality oracle.
7. Match timing starts/ends by `(phase, thread, started_monotonic)`; keep
   inclusive wall and thread-CPU values. Per-request helper windows select
   contained worker spans. Three candidate-pin checks must remain outside
   the one HTTP helper span. Nested/concurrent totals are non-additive.
8. Logical zero, clean close, ECHILD and idle are independent observations;
   they do not fill unobserved physical-admission, allocator or release fields.

`manifest.json` records compressed and uncompressed content hashes and sizes.
`verify.py` validates them, rederives all published scalars and checks the
sealed scalar projections. `--check-source` additionally rehashes every
original file and the separately listed plan/history inputs. No GPU or
third-party package is needed to inspect or verify the bundle.
