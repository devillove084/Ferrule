# Evidence types and file map

All JSON is UTF-8, rejects duplicate keys/nonfinite numbers, and preserves JSON
`null` for unavailable measurements. Booleans are not accepted as integer counts.
The checker is deliberately specific to this frozen four-cell cut, not a general
telemetry framework or a statistical fault-reliability claim.

## Files

| File | Meaning |
|---|---|
| `evidence/mixed_clean.jsonl`, `evidence/mixed_unknown.jsonl` | Actual structured producer records, 40 rows each |
| `evidence/*.stdout` | Exactly one selected test PASS per process; no selector retries |
| `evidence/*.stderr` | Matching 40 `COMPONENT_FAULT` records per process; no extra text |
| `report-contract.json` | Reviewed closed vocabulary, fixture/header, action sequences and seal |
| `postfailure.json` | Deterministic typed projection derived from actual records |
| `custody.json` | Safe ownership, ECHILD, lock and 26-sample NVML projection |
| `history.json` | Preparation versus actual execution, plus preserved scope limitations |
| `source-manifest.json` | Source/build/report identities; full relative source pin inventory |
| `manifest.json` | SHA256 and size of the 19 curated data/reference payload files |
| `reference/*.rs` | Nine byte-identical pinned source files needed to read the assertions/seam |
| `figure.json` | Checked plot input, renderer/source/image pins, measured text bounds |
| `evidence.py`, `archive.py`, `check.py` | Parser/invariants, optional private curator, portable check |
| `test_publication.py`, `test_figure.py` | Stdlib-only generative negative tests |

The manifest covers the generated payload, not its own bytes or the checker.
The figure has a separate manifest to avoid a cyclic data/figure hash. Versioned
source and these explicit trust anchors are intended for reviewer inspection.

## Measurements are not inferred API receipts

`postfailure.json` separates:

- **`snapshots`**: serialized field measurements of committed metadata, owner
  charge/custody, live-state byte count, active transactions, free KV slots,
  host charge and physical-observation flags. Snapshot sampling is passive;
  it does **not** read KV or recurrent device contents. `committed_page_slot`
  is an observed pool slot, not a fabricated per-page reference count.
- **`boundary`**: serialized test-hook observations of the real routed-down
  submission, successful actual compute-event/upload wait, and whether its
  proof was deliberately discarded. `error_kind` is a serialized classification
  backed by the test's typed downcast, not a raw error object.
- **`tx2_counts` / `host_read_statistics`**: serialized counter deltas, checked
  against before/after counters where available. CUDA totals cover the whole
  tx2 owner, not individual read tickets. Wasted bytes are retained, not hidden.
- **`committed_content`**: clean `equality: true` is a **serialized assertion
  result**, backed by pinned bitwise-comparison source plus PASS. Raw state bits
  are absent. Unknown `equality`, `raw_contents` and byte-content fields are
  **`null`**, with an explicit unobserved basis—not `true` or zero.
- **`assertion_backed`**: assertions inferred from the pinned test actions and
  successful outcome. Two independent compute and two upload waits, two typed
  Unknown aborts, rejection of reset/admission/KV release/publish/shutdown, and
  successful clean tx3 are **not separately serialized per-call result receipts**.
  `standalone_wait_receipts` and `individual_error_payloads` are `null`.
  The rejected reset/publish/etc. calls assert `is_err()`; only the execution and
  abort paths assert typed `CompletionUnknown`. Do not claim every rejected API
  serialized or downcast the same error type.
- **`local_final_release`**: only clean records an explicit allocator observation
  after owner drop and retirement trim. It is distinct from clean-abort snapshots
  and not a device-wide/global release claim.
- **`unknown_retention`**: records deliberate retention until process exit;
  no forced allocator zero, release certificate, or in-process refund.

### Exact source/row navigation

In each 40-row JSONL, Blocking occupies 1-based lines **2–20**, Lookahead **21–39**;
line 1 is the header and line 40 the final seal. Fault boundaries are lines
**10 and 29**, cell outcomes **20 and 39**. `postfailure.json` gives the individual
snapshot line numbers. In clean, content checks/tx3 are in
`reference/fault_cuda.rs` lines **424–493**; Unknown fences/abort/rejected methods
are lines **494–581**. `fault_hook.rs` captures the actual submission/physical
proof, and `residency.rs` implements successful real waits **before** test-only
proof discard. Source lines refer to the archived bytes, not a moving checkout.

## Non-additivity and release limits

`cache` fields obey:

```text
resident + pending + scratch = charged
free + charged = budget = 65536
0 <= unknown_subset <= charged
```

The 6,918 B Unknown quantity is **inside** charged: 774 B pending plus 6,144 B
scratch, including the persistent reservation. Clean has 4,096 B scratch and
no pending/resident bytes at the post-failure/cleanup boundary.

`recurrent_workspace_bytes_separate = 318720` names a separate overlapping owner
view. `workspace_overlap_bytes`, `aggregate_physical_retained_bytes`,
`exact_kv_reference_counts` and `state_content_bytes` remain `null`.
No disjoint physical sum, exact KV refcount, or post-Unknown bitwise proof exists.

`custody.json` reports authenticated drain/audit conclusions after removing
private identity and authentication material. Its GPU quiescence/release fields
remain unverified/unobserved. NVML idle and ECHILD do not repair Unknown or grant
physical release authority. Two utilization-only transient samples are retained
as **unaccepted**, not erased or relabelled idle.
