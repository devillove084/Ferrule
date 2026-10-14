# StorageMiddle64 repeated-QA / NFS-client pilot v2

**CPU preparation only; no GPU authorization, query, lock, server launch, HTTP,
new measurement, or scientific PASS.** This additive version does not edit or
execute historical `case-middle.json`, `run_middle.py`, `cpu-revisions/`, `runs/`,
manuscript/production code, or existing analysis/plot generators.

The sealed preregistration has a **new ceiling of 12**, explicitly resolving
`4 initial + 1 excluded-first QA + 2 warmups + 5 measured QA = 12`, not 10.
Five fresh server lifecycles `L1,L2,L3,L4,L5` use the identical new candidate,
`StorageMiddle64` effective config, selected physical UUID and request order.
No retry, replacement trial, small/big point, power calculation, significance,
CI, steady-state, comparative-speedup, or storage-cold claim is preregistered.
New processes are independent lifecycles, not independent page-cache/NFS samples.

## Fixed budget before any future execution

| Bound | Literal seconds |
|---|---:|
| Startup from launch | 700 |
| One HTTP request | 120 or less remaining aggregate/lifecycle budget |
| All 12 HTTP requests, absolute non-sliding aggregate deadline | 480 |
| Target bookkeeping/observation reserve | 60 |
| Target lifetime, inclusive of startup + HTTP + bookkeeping | 1240 |
| Target normal direct-child shutdown, additional to lifetime | 45 |
| Worker pre-idle / finalization | 20 / 35 |
| Supervisor application timeout | 1340 |
| Supervisor public-call worst-case overhead reserve | 28 |
| Coordinator pre-idle / post-idle / finalization | 20 / 20 / 32 |
| Outer lifecycle | **1440 = 24 min** |
| Entire five-lifecycle pilot | **7200 = 2 h** |

Checks: `700+480+60=1240`; `20+1240+45+35=1340`;
`20+1340+28+20+32=1440`; `5*1440=7200`.
The maintained `run_target` accepts lifetime <=1560 and shutdown <=45.
`run_process` reserves a 5s handshake, 5s command send, 8s helper wait grace,
8s cancellation wait and 2s direct-helper reap, conservatively summed as28s.
The outer deadline closes admission; authenticated cleanup retains authority.
An OS/syscall that cannot be interrupted is not a promise of physical completion:
record overrun/unknown custody and never begin the next lifecycle.

`480 < 12*120` is intentional aggregate censoring, not contradictory independent
120s allowances. Each request gets `min(120, remaining480, remaining target,
remaining outer, remaining pilot)`; if no time remains it is unissued. Startup,
request, sampler, parser and shutdown work must share those absolute monotonic
deadlines. They cannot reset on progress. Capture/JSON/observer costs consume
reserves; stop rather than extend the budget. The helpers do not enforce the
future worker's timers themselves and expose **no launch entry**.

Planning source is **historical n=1**, `-02/canary-qa/http.json` E2E
18.309471419081092s (startup18.497s). All60 requests may be as expensive as QA:
`60*18.309471419081092=1098.5682851448655s`. Add a deliberately conservative
500s/lifecycle startup/idle/observation/shutdown scheduling allowance: total
3598.5682851448655s, **about1h planning**, not measured current-binary cost.
The fixed worst-case cap remains2h; failures never cause extra trials.

## Preregistered order, outcomes and denominators

In each lifecycle: Hello2, QA8, arithmetic8, Chinese8, QA-first-excluded8,
QA-warm1/2, QA-measured1..5. Seed1, temperature0, one choice, serial concurrency1,
raw streaming completion with usage; context128. The excluded-first QA occurs
**after four requests**, not first-after-server-Ready. All QA gates compare
actual eight-ID enqueue history, frozen text and usage. Hello has no frozen
numerical reference; it is an HTTP/usage/trace-consistency canary only.

Primary outcomes: client content-event TTFT and E2E, per-request logical
checkpoint read calls/bytes, expert hits/uploads/evictions, facade H2D/artifact
copies/bytes, sampled GPU/tree PSS high water and raw shutdown-summary coverage.
Keep startup logical counters separately. `hits/(hits+uploads)` is only a
named lookup/upload share with a nonzero denominator, not a universal cache hit
rate; upload is not a proven miss counter. Reader/syscall-level bytes are not
physical network/disk bytes; facade H2D is not PCIe traffic. PSS is not a physical
host cap and excludes unmapped page cache; samples are not exact peaks.

Denominators remain **5 lifecycles / 60 scheduled requests / 25 measured QA**.
For each lifecycle/phase: scheduled, issued, success, failed, censored, unissued;
`issued=success+failed+censored`, `terminal=success+failed`, and
`scheduled=issued+unissued`. A journaled send with lost terminal evidence is
censored, never unissued. All missing future slots stay unissued after STOP.
HTTP success is separate from quality/metric eligibility. Only explicitly
eligible successes enter descriptive latency rows; display denominators and
failure/censoring counts alongside them. No automatic significance analysis.
Partial HTTP data is a lower bound; tokens after cutoff do not turn a censored
request into a successful sample. Raw actual IDs, not retokenized visible text,
are required for an exact-token PASS. Synthetic tests are not run evidence.

## Existing maintained APIs, not a second harness

Reuse `contract.py` for pins/exclusive receipts/staged descriptors/local UUID
locks/drain, `launch_guard.seal/authenticate` for coordinator/helper/worker
PID+birth+parent binding, `_owned_subreaper.py`/`run_process` for authenticated
custody/ECHILD, `_profile_capture.run_target(...,1240,45)` for finite direct-child
wait/TERM, `observer.Monitor/verify_root/idle`, and `bench_ferrule_vllm.main` for
bounded HTTP. A future worker must be a specifically sealed **TP1** adapter,
not `experiment_current.py`'s fixed TP2 dispatcher or the frozen four-request
middle wrapper. This change adds evidence/preparation adapters only; it does
not install a generic supervisor, launch framework or GPU-capable CLI.

Future coordinator registration must bind the final-candidate hash, five trial
IDs,12-request ceiling, entire order, actual capacity/UUID, exact argv/env,
all deadline literals, wrapper/helper pins and worker nonce. The held UUID
`flock` is **local nonblocking serialization, not global GPU ownership or an
NFS-client reservation**. Require renewed sole-owner consent, selected physical
UUID/ordinal0, birth-bound roots, pre/post all-eight idle, nonselected positive
memory delta tolerance1MiB, and unchanged loaded-library mappings. Never signal
foreign PIDs, kill by name/group, drop page cache, install packages, or touch
frozen runs. Existing outer authenticated custodian alone handles descendants.

## Trace and resource summary capture

Opt in to JSON logs with:

```text
FERRULE_LOG=info,ferrule_performance=debug,ferrule_request=debug,ferrule_token=trace,ferrule_resource_observability=info
FERRULE_LOG_FORMAT=json
```

Archive raw `server.log` after normal shutdown and log drain, preserving hashes,
line/byte offsets and monotonic+wall request boundaries. Log only approved
public canaries; IDs/SSE reconstruct model output and mountstats reveal export
paths. Restrict artifact access and do not upload externally. DEBUG/TRACE and
capture overhead is unmeasured; this is an instrumented baseline, not a fast
uninstrumented benchmark.

`request_worker_token_enqueued` carries actual `request_id`, `session_id`,
`emission_ordinal` (assigned only after successful channel enqueue),
`actual_token_id`, `resident_token_index`, and `worker_channel_enqueue` boundary.
Join to **actual raw SSE JSON completion IDs** `cmpl-ferrule-N`, not client
request order/name, resident index or invocation transaction. Preserve opaque
IDs as unbound; never strip arbitrary digits or guess an ID. Reject duplicates,
gaps, noncontiguous enqueue order, reused request IDs, missing trace and usage
mismatch. Archive `request_worker_terminal_enqueue` and `request_http_terminal`
for terminal enqueue/HTTP-side counts separately. Client nonempty content-event
ordinals/timestamps are independent of token enqueue ordinals; one SSE chunk
may contain multiple tokens, hidden/empty tokens, or no visible content. Worker
enqueue is not HTTP consumption, socket delivery, or exact client token timing.

Parse every `resource_shutdown_summary.summary_json`, not fabricated legacy
`*_observed` closure fields. Retain `schema_version/attempt/outcome/error`,
`runtime.{scope,resident_bytes,pending_operations,active_credit_grants,
pending_retirements,fence,closure,quarantine}`, `physical.{scope,residency,
quiescence,close,release,expert_cache,allocator}`, broker units and coverage.
Evidence `{state: observed|verified|unobserved|unknown, value: ...}` qualifiers
stay intact. Any explicit Unknown in an attempt stays sticky through later
attempts. `complete` means an explicit owner-local close, not post-Drop release.

**STOP on unknown custody**: lost PID/birth/parent/UUID or authenticated drain,
foreign GPU owner, explicit unknown owner/scope/fence/closure/quarantine. Pending
final shutdown, missing required close identity/quiescence, request/quality/log/
mount reset failures also STOP with distinct failure reasons. Unsupported
resource partitions or final release being `unobserved` alone are **not** unknown
physical custody. An actual mapped retained owner with complete/closed/quiescent
scope, observed zero pending/unknown cache bytes and no quarantine may proceed
only after normal exit, authenticated drain and independent idle. Retained
source mappings/allocator residency are legal; **do not require universal final
release or zero retained residency**. The helper reports
`resource_contract_complete=unverified`, never invents a universal `verified`.

## Passive NFS-client windows and claim scope

Use `capture_mountstats` only inside the authenticated future worker. Supply
maintained `observer.verify_root` and its **worker/sampler PID+birth+parent** so
the reader remains alive for post-server-exit samples. Bind reader/server mount
namespace at Ready and seal the namespace/boot ID. No active NFS probe, checkpoint
payload read, cache manipulation, external traffic suppression, or foreign
process control is permitted.

Capture raw text and JSON with read-begin/read-end monotonic+Unix timestamps,
source `/proc/<reader-pid>/mountstats`, identity, boot ID, mount namespace,
export/device/mountpoint/fstype/statvers, mount age, raw SHA and labels:
`prelaunch`, `ready`, `requests-pre`, each `<request>-pre/post`,
`requests-post`, `shutdown-pre`, `shutdown-post`. Startup/request/shutdown
windows are separate. Longest NFS mount must cover **all** checkpoint paths;
no ambiguous overlay or duplicate mount is silently selected. Store an explicit
error/null capture if unavailable; do not infer zero or fall back to `/proc/io`.
A window brackets client sampling and may include unrelated traffic; it is not
an exact syscall-time or exclusively owned physical-IO interval.

Only stable boot/mount namespace/device/mount/statvers/age and nondecreasing
counters yield deltas. Counter decrease, age reset, missing data, malformed
layout or namespace drift suppress deltas, retain raw snapshots, mark reset/
missing, and stop further pilot issuance. Track the eight NFS `bytes:` counters
and READ RPC counters separately; `serverreadbytes` is a **shared client NFS
counter**, not server-attributable or NAS backend disk bytes. PID `read_bytes`
is not NFS network traffic. Physical NAS backend/disk IO and foreign traffic
remain unknown, even if the selected GPU is locally idle/exclusive. No claim
of storage necessity, page-cache coldness or physical read attribution follows.

## New baseline and quality prerequisites

`new-baseline-build-requirements.json` records literal maintained build argv/env
for a new GPU-hidden offline/locked CUDA-only opt3/fat-LTO/codegen1 release CLI,
explicit `sm_86`; do not run this build as part of CPU preparation. A new release
must include current token/request/summary sinks, current production/native
source/archive relation, Rust/compiler/CUDA/CUTLASS/runtime pins, immutable
staged executable, exact-binary CPU-hidden help/serve-help and actual loaded
library mappings. Same historical argv controls do not make the old binary
qualified. No historical -02 or older full40/fresh PASS is current acceptance.

GPU owner must first pass these actual current `StorageMiddle64` selectors:

- `model_factory::dense_bf16_tests::full40::numeric_bf16_mixed_storage_middle64_public_factory_full40_logits_routes_states_match_profile_reference`
- `model_factory::dense_bf16_tests::full40::fresh_quality::mixed_storage_middle64_fresh_quality_retained_generation`
- `model_factory::dense_bf16_tests::full40::fresh_quality::mixed_storage_middle64_fresh_quality_cold_owner_restart`

The cold receipt binds its primary fresh report/reference/config. Archive
actual schemas `ferrule.numeric-bf16-full40-profile-gate.v1` and
`ferrule.bf16-fresh-quality-result.v1`, with `passed=true`, `complete=true`,
full effective config, independent-reference pins, exact fresh IDs, empty
failures, normal shutdown/no quarantine and authenticated custody. The terminal-join helper separately reports terminal_reconciled and suppresses
metric_eligible without both the worker-terminal enqueue and HTTP-side
terminal. Source-level
test results need a pinned **compiled production source relation to the new CLI
binary**, not an invented same-binary assertion. Later `bind-candidate` only
validates/seals these CPU-readable receipts, never launches a process/GPU.

Candidate input is a new JSON receipt `schema=ferrule.storage-demand.candidate-binding.v2`,
`status=quality_passed_cpu_bind_requested`, `protocol_sha256`, `binary`,
`binary_sha256`, `build_receipt`, `runtime_pin_manifest` (`pins` and
`unknown_nccl_enabled=false`), `source_archive`, `source_archive_manifest`,
`serve_help_receipt` (`status=pass`, `gpu_hidden=true`, exact binary SHA and
`argv_flags`), `full40`, `fresh`, `cold`, `quality_source_relation`, and absolute
`artifact_pins` for every item. The source relation records CLI SHA,
`production_source_digest`, per-report SHA, cold primary SHA, independent
reference evidence, supervisor_receipts, test_source_manifests, and
test_executables maps for full40/fresh/cold, plus reference_manifests maps
for all three. Pin every referenced artifact. The gate checks actual
contract.successful_process supervisor receipts, matching source manifests
and reference hashes; declaration strings alone cannot pass. It is evidence
metadata, not a new cleanup certificate. Final candidate seal cannot be issued now:
no new release/linked-quality receipt has been supplied. Fixed protocol is
preregistered first, candidate later, and **neither is GPU authorization**.

## CPU-only commands (from repository root)

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/tests -v
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/newversionedprotocol/protocol.py plan
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/newversionedprotocol/protocol.py check
```

`seal-cpu` is the one-time exclusive-create authoring operation for this version;
after the seal, `check` is read-only. Changed protocol/helper/docs/test pins need
a **new version**, never automatic resealing or edits to historical receipts.
`bind-candidate --candidate <owner-supplied-new-receipt> --output <new-file-in-newversionedprotocol>`
is intentionally not a runnable example with invented paths: obtain literal
sealed artifact paths from the GPU/build owner first. There is **no run/GPU
mode**, and no execution command is offered in this CPU-only handoff.
