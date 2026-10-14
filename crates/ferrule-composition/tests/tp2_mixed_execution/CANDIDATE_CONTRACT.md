# Mixed worker candidate capture v2 — bounded diagnostic acceptance

The capture is disabled by default. No CLI/catalog Mixed capability is enabled.
Only `ParallelExecutionPlan::start_internal_with_trace(..., Some(MixedTraceConfig))`
installs the model's native owner-local diagnostic observer. The same executor,
sealed collective, state journal, KV commit, and preparation claim still run.
Model/backend/paper sources are not modified by this integration.

## Exact selectors (composition `tp2_mixed_execution`, `--features cuda`)

- `full40_candidate_boundaries_two_cases_two_phases`
- `lifecycle::tiny_prepared_cancel_multistep_and_recreation`
- `lifecycle::full40_bounded_multistep_fork_recreate_cold_owner`

All three are ignored GPU tests. Compile with GPUs hidden. Execute only after
new GPU-owner permission, through authenticated bounded supervision, one exact
selector per invocation (`--exact --ignored --nocapture --test-threads=1`).
Do not add them to the ordinary CPU test command via `--include-ignored`.
The existing launcher selector allowlist must be updated by the launch owner;
no new supervisor is provided here.

Keep CUDA12.4/sm86, physical GPU0/1, `CUBLAS_WORKSPACE_CONFIG=:0:0`, 22 GiB/card,
`FERRULE_MIXED_GPU_RUN=1`, original checkpoint and original oracle unchanged.
Capture requires `FERRULE_MIXED_EVIDENCE_DIR` (fresh directory),
`FERRULE_MIXED_MODEL`, `FERRULE_MIXED_ORACLE_DIR`, and
`FERRULE_MIXED_ORACLE_SHA256=fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d`.
The test authenticates the original oracle/checkpoint before CUDA startup.

## On-disk candidate contract for the independent comparator owner

`EVIDENCE/candidate/contract.json` declares `ferrule.mixed-candidate.v2`.
Per owner: `candidate/rank-{0,1}/events.jsonl`, `tensors.f32le`, and
`capture-status.json` (`ferrule.mixed-owner-trace.v2`). All files are exclusive-create.
The candidate schema is independent of the unchanged runtime process wire.

Events:

1. `owner_binding`: authenticated plan epoch/group/source/ordered members;
   bounded debug descriptions of data/route sites, plus rank lane/layers/precision
   and residency mode. Structured tensor partition fields below carry geometry.
2. `begin`: monotonically numbered owner-local invocation; actual transaction,
   session IDs, tokens, positions, sequence phases/context/query ranges,
   page/execution generations and state indices from the validated packed batch.
3. `tensor`: `invocation`, owner-global `order`, `event`, global `layer`, byte
   `offset` and `bytes` into `tensors.f32le`, `storage=f32le-native-host-view`.
   `native_dtype` is `Bf16` or `F32`; `shape` and `strides` are native N-D geometry
   in elements, not export-byte strides. BF16 is downloaded as native words and
   expanded exactly on the host for serialization only. No GPU F32 expansion.
   Optional `sequence` contains packed row/position indices and input frontier.
   `parameter_partition` contains checkpoint row ranges, columns and collective;
   it is not necessarily the activation shard axis. `attention_partition` is a
   structured `gqa` or `gated_delta` object with head/channel ranges, replica owners
   or GDN shape. All ranges are half-open. Parameter path/role identify bindings.
   `router` carries integer `ids` and authoritative raw-u32 F32 `weight_bits`.
   `router.probabilities` carries the model's actual native F32 full probability
   source, not a runtime softmax reconstruction. Operator events are provisional:
   **`committed=false`, including GDN working-state observations**.
4. `execution_end`: execution success/failure only, always `committed=false`.
5. `committed_kv_begin`: transaction, logical source owner, state slot, generation,
   incarnation, committed token count and page size, after publication/retirement.
   Subsequent `tensor` events named `committed.kv` have `committed=true`, global and
   local KV layer, plane (0=K, 1=V), positions, shape `[tokens,local_kv_heads,head_dim]`,
   strides, KV head range/replica owners, physical page/slot/epoch and logical source
   identity. These chunks cover the retained prefix plus append, not only new rows.

Join `candidate/{hello,capital}.{prefill,decode.0}.receipt.json` with owner `begin`
by exact transaction/session: only a successful parent `published=true` receipt
establishes execution publication. Join committed KV additionally with
`{case}.{phase}.kv-receipt.json` (`ferrule.mixed-committed-kv-receipt.v1`) and require
`export_completed=true`, matching transaction, and **both owner writer statuses**.
Export completion proves source-lease/readback completion, not filesystem drain.
Partial KV records before a failed export are not a complete snapshot.

The parent acquires and validates the actual page-manager committed read lease
before and after each owner export. Owners require the exact retired publication,
matching generation/frontier, no active custody, and quiescence. Local KV geometry
comes from the same sealed rank plan; only schema is projected from global heads.
Unknown retains the registered source pin and quarantines the pipeline; ordinary
fenced failure releases the pin. Export also requires the owner writer's latest
captured transaction: A published, then B executed/cancelled, then export A is
rejected before readback rather than mislabelled with B's invocation. No provisional append or reconstructed source
is accepted as a post-publication snapshot.

Existing full-vocabulary `.actual.f32le` and `.metrics.json` files remain at
EVIDENCE root. The original tolerance and oracle hash are unchanged. A numerical
failure stops the test; missing/failed receipts cannot become committed evidence.

## Bounded capture / admission

Per owner: **unchanged** 1 GiB tensor total, 40,000 tensor events, four invocations,
8 MiB exported tensor cap, 64 MiB total serialized index, 256 KiB per index record.
`queue_record_limit()` derives 40,013 slots: all tensor events plus one binding and
three control records per invocation. There is no independent four-slot pressure
limit or assumption that the filesystem keeps up. The writer may lag the whole
capture. Payload and metadata are exact boxed slices (not per-record cap-sized
allocations), charged against their separate total byte caps. Draining does not
reset the total capture counters. First failure records `reason`, `event`,
`attempted`, `limit`, `pending_records` and `pending_bytes` in `capture_failure`;
`capture_error` retains a specific reason, and `peak_pending_bytes` is reported.

Owners still use **only try_send**. No sleep, filesystem operation, disk wait or
collective backpressure is introduced. A full derived channel is an invariant
failure, not a throughput policy. Oversize tensor, invocation, total bytes/events,
metadata limits, disconnected writer and writer I/O failures are distinct,
fail-closed causes. The parent waits at most ten seconds after owner shutdown.
On failure/unwind `abort_capture` independently stops `recv_timeout` within its
25 ms polling interval, even when Unknown retains the sender. No device custody
or logical read lease is released by host abort. A disk syscall itself cannot be
cancelled by the thread; parent wait remains bounded and incomplete on timeout.

Assembly and direct preparation use the same `host_reserve_with_pages` and
`validate_admission`. Reserve per owner:

```
max_bytes_per_owner + max_metadata_bytes_per_owner
+ queue_record_limit * (sizeof(Record) + 6*sizeof(usize))
+ max_tensor_bytes + CudaDiagnosticLimits::host_tensor_bytes()
+ 2 MiB bounded metadata/serialization working storage
+ 2 MiB writer stack
+ physical_pages * (6*sizeof(KvPageId) + sizeof((KvPageId,u32,u64)))
```

The slot term includes channel stamp/alignment and per-record allocation-header
allowance. Pending payload/index storage is admitted for the **whole capture**,
including the in-flight writer, plus independent download/serialization peak.
The test retains 256 MiB execution host allowance and explicitly adds this full
capture claim under the existing 128 GiB supervised host target; it does not
pretend the previous 256 MiB allowance can hold two 1 GiB captures. Direct prepare
rejects missing/insufficient allowances and rechecks mutations after admission.
Device scratch is still model allocator-rounded 8 MiB at the chosen cap; native
BF16 export has no device expansion or pinned staging. **GPU22 GiB/card remains
unchanged**, as do numerical tolerances, oracle identity and all inventory rules.

## Required inventory is independently decided

`complete_available_trace` means only this bounded native stream drained intact.
`required_inventory_complete` remains **false**. Full40-03 required fields, shard
concatenation, replica consistency, cache chunk coverage, route comparisons, and
canonical archive mapping are the independent comparator owner's responsibility.
This version supplies native probabilities/geometry and a post-publication KV
bridge. Full40-03 and the released lifecycle recipes have completed independent
inventory comparisons; receipts and limitations are recorded below. GDN state remains a provisional operator observation joined
to publication, not a separate post-publication GDN snapshot. No missing event may
be silently substituted, recomputed or accepted by loosening the contract.


## Pinned lifecycle recipe producer

`lifecycle::full40_bounded_multistep_fork_recreate_cold_owner` now consumes one
completed, SHA256-pinned paper lifecycle reference rather than a self-consistency
script. The name is retained for launcher continuity; scenario comes from the
reference `suite.plan`. `lifecycle_plans.json` admits exactly the frozen release
plans: greedy4 (10 calls), fork2 (6), canary2 (5), recreate1 (2+2), and cold-owner1
(4 calls per independent process). It binds release SHA256
`834c8a1ff0aa12d0e58e500e870e68950a2dbe4e3cb6902d47fe02fdcd459639`
and contract SHA256
`684dd2f85798745178a3a6719daaf23cc44471072fb3c35e3155065fb8ba170d`.
Changed prompts, truncated/reordered operations, relabelled recipes and widened
limits reject before GPU initialization; no default-step substitution occurs.

Required supervised environment:

- `FERRULE_MIXED_LIFECYCLE_ORACLE_DIR`: completed scenario reference directory
- `FERRULE_MIXED_LIFECYCLE_ORACLE_SHA256`: its manifest SHA256
- Existing `FERRULE_MIXED_MODEL`, fresh `FERRULE_MIXED_EVIDENCE_DIR`, owner grant,
  CUDA12.4/sm86, GPU0/1, `CUBLAS_WORKSPACE_CONFIG=:0:0` constraints still apply.
- `FERRULE_MIXED_ORACLE_DIR` is not substituted for the lifecycle reference.

Each call replays recorded reference inputs, including greedy tokens even if the
candidate would predict differently. Actual full-vocabulary logits are written
and checked with the original tolerance. Create/fork/drop/recreate use the same
executor, journal and claim-aware constructor. Recreate shuts down and drops the
old image before initializing another. Cold-owner is **one fresh supervised
process per launch**; a distinct cold-process proof requires a second authorized
launch, not an in-process loop. The separate tiny prepared-cancel test remains;
extra cancellation operations are not inserted into a pinned numerical recipe.

Raw schema: `ferrule.mixed-lifecycle-raw.v1` in `EVIDENCE/lifecycle-raw/`:

- `contract.json`: reference pin/checkpoint/protocol, exact plan, epoch call counts,
  fixed pipeline config and complete capture bounds.
- `epoch-N/`: `ferrule.mixed-lifecycle-owner-capture.v1` contract plus the unchanged
  v2 rank event/payload/status streams. Transaction identity is scoped by epoch.
- `epoch-N/call-NN.receipt.json` and `.kv-receipt.json`: actual parent publication
  and retained-source KV completion. No receipt is synthesized from reference.
- `operation-NN.json` and `result.json`: executed operation order, logical/runtime
  session mapping, frontier, epoch, transaction and actual next-token observation.
- `EVIDENCE/call-NN.call.actual.f32le` and `.metrics.json`: actual full logits.

Raw completion is `status=captured`, NOT canonical candidate/quality completion.

### CPU canonical adapter (composition tests; paper sources unchanged)

`lifecycle_bridge.py` reuses the paper's existing `read_owner`, `join_call`,
`assemble_call`, tensor digest function and `validate_suite`. It does not run a
model, compute router probabilities, sum partials or fill missing state. Bounded
one-call temporary projections only copy actual raw bytes and normalize
invocation/order/offset numbers for the existing strict parser. On-disk originals
and reference artifacts are immutable, SHA checked before and after conversion.

Canonical output is exactly `ferrule.paper.tp2-mixed-lifecycle-candidate.v1` with
`cases == suite.cases`, exact `suite.plan`, ordered `suite.events`, publication
file hashes, one `call` archive/report per call, and actual state hashes:

- `state_after`: post-execution GDN observations joined to successful publication;
  leased post-publication GQA KV exports (same semantics as the proven full40 bridge).
- `state_before`: actual `gdn.before.conv` and `gdn.before.ssm` observations;
  the retained prefix sliced from the **actual committed KV snapshot**, never from
  projected new K/V. Every continuation/fork must match its preceding candidate
  state hashes. Initial empty GQA prefix hashes derive from a zero-length slice.
- fork digest: the source's preceding captured publication, verified against the
  branch's actual before-state on the next call; not copied from reference.
- untouched session list and epoch/session continuity checked by the maintained
  paper lifecycle validator. This is not new physical cancellation/COW proof.

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /opt/conda310/bin/python -B \
 crates/ferrule-composition/tests/tp2_mixed_execution/lifecycle_bridge.py \
 --input FRESH_EVIDENCE --inspect

# Supply the inspect hash, actual reference pin, and a fresh output directory:
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /opt/conda310/bin/python -B \
 crates/ferrule-composition/tests/tp2_mixed_execution/lifecycle_bridge.py \
 --input FRESH_EVIDENCE --input-sha256 RAW_SHA \
 --reference LIFECYCLE_REFERENCE --reference-sha256 REFERENCE_SHA \
 --output FRESH_CANONICAL
```

Then run the unchanged paper `lifecycle_compare.py` command from `BRIDGE.md`.
Conversion success is not a numerical PASS. It is not the legacy two-case
`bridge.py --input` CLI; that CLI intentionally remains fixed to full40-03.

### Single fresh run bounds

All scenarios: page size 16, logical pages 32, physical pages 64 per owner,
max rows 8, max context 128, sessions 3, state copies 7, GPU 22 GiB/card.
Capture total tensor budget is **4 GiB across both owners and all epochs**,
8 MiB per tensor, 64 MiB index per owner/epoch. Event allowance is an explicit
10,000 per call (at most the existing 100,000 owner cap), not a throughput queue.
Every queued byte/record is charged by the existing trace admission formula.

| Scenario | Calls / epoch | Image loads | Events / owner / epoch | Host allowance including 256 MiB execution |
| --- | --- | --- | --- | --- |
| greedy4 | 10 | 1 | 100,000 | 4,765,562,688 bytes |
| fork2 | 6 | 1 | 60,000 | 4,758,520,588 bytes |
| canary2 | 5 | 1 | 50,000 | 4,756,760,058 bytes |
| recreate1 | 2 + 2 | 2 sequential | 20,000 | 2,603,994,832 bytes peak |
| cold-owner1 | 4 | 1 per launch | 40,000 | 4,754,999,536 bytes |

CPU real-header estimation for this exact session capacity returned per card:
weights 18,390,764,032; seven-state grants 227,082,240; KV 12,582,912;
complete device ledger with trace **19,427,749,888 bytes**, below 23,622,320,128.
Base known host ledger is 89,187,512 bytes. These are admission estimates, not
measured GPU/driver/PSS peaks; startup re-estimates and rejects unsupported shapes.

### Scope / blockers

- The released recipes have GPU/CPU evidence below. Further GPU execution still
  requires a new owner grant; these results do not enable the public capability.
- Shared-prefix read authority remains strict. A fork alone cannot export an
  aliased source. In the fixed fork recipe, Hello's one-token partial page is
  COWed by each branch append, enabling a legal active-branch export. The CPU
  regression verifies that path and verifies a full shared prefix page still
  rejects after append. Longer-prefix recipes cannot bypass this; they remain
  unsupported until a real read-authority capability exists.
- GDN publication binding uses real provisional observations, not a new
  post-publication GDN API. Missing before/after/router/KV data rejects conversion.
- Only the five exact released plans are admitted. Other step counts still need
  explicit fixture/budget support, never a silently widened admission.
- Cold-process proof, physical untouched-session sampling, mid-compute failure
  and numerical cancellation rollback are not implied by lifecycle comparison.
- Public capability, CLI, production mathematics and paper sources are unchanged.


### Released lifecycle evidence (2026-10-09)

Evidence root: `target/validation/tp2-mixed-lifecycle-release-gates/`.
The unchanged paper comparator outputs are
`papers/arxiv/experiments/tp2_mixed/oracle/runs/runtime-lifecycle-RECIPE-comparison-01.json`.
All six comparisons pass: 33 calls, 35,937 required tensor metrics, every metric
exact (`max_abs=0`, failures=0), including captured routing, state and TP shards.
Original tolerance remains `0.00025 + 0.012 * abs(expected)`.

| Recipe | Physical capture | Calls | Comparator |
| --- | --- | ---: | --- |
| greedy4 | `greedy4-01` | 10 | pass; supervisor pin-audit caveat below |
| fork2 | `fork2-03` | 6 | pass |
| canary2 | `canary2-01` | 5 | pass |
| recreate1 | `recreate1-01` | 4 | pass; two sequential owner epochs |
| cold-owner1-01 | `cold-owner1-01` | 4 | pass; independent launch |
| cold-owner1-02 | `cold-owner1-02` | 4 | pass; second independent launch |

The CPU adapter preserves the pinned reference's prompt-object representation
before the existing suite validator: Rust sorts JSON object keys, while the paper
plan generator iterates prompt insertion order. Raw/reference plans must still
be equal; operation arrays are never reordered. Sorted-key acceptance and changed
operation rejection have dedicated CPU regressions.

Greedy's original GPU test passed and custody drained, but its supervisor remains
`passed=false`: it unnecessarily pinned unused `tp2_resident/paired.py`, which
changed during execution. `greedy4-01-recovery-audit.json` records dependency and
source/binary verification separately; the failed receipt is not rewritten.
Future launches pin the actual observer/sampling dependency closure.

`fork2-02` failed at initial `cuInit` before model execution. Its failure is retained.
An authenticated initialization-only probe then passed with the same environment;
the one fresh retry `fork2-03` passed. The initialization failure's root cause is
not established; historical kernel logs are not evidence of a current GPU fault.

All completed captures used CUDA12.4/sm86, GPU0/1, `:0:0`, and 22 GiB/card.
Allocator peak reserved bytes were 19,086,180,352 (19,153,289,216 for fork/canary).
Sampled device peaks were 19,372,441,600 (19,439,550,464 for fork/canary), below
22 GiB; these are sampled device-wide observations, not exact process-attributed
peaks. Host tree PSS reached 53.36–53.44 GB, or 76.73 GB for recreation, below the
128 GiB supervised host limit. The capture allowance above is not a bound on total
startup/checkpoint/process memory. Sampling gaps remain disclosed in each receipt.

Claims finish before execution; shutdown fences report quiescent/no quarantine.
Shutdown logs still include the intentionally retained image; they do not prove
zero live allocator bytes at that point. ECHILD and all-eight idle after process
exit prove final process cleanup. Summary/pins/identities: `acceptance-summary.json`.
No vLLM equivalence, performance result, public Mixed admission, cancellation
numerical quality, or arbitrary-prefix KV export capability is inferred.
