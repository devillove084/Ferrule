# Cache-on release: CPU preparation, not GPU authorization

> Historical dadf release preparation below is retained as written. The newer
> dedicated-uninitialized root-arena **internal numerical** closeout is
> `runs/decoded-arena-uninitialized-20261010-01/HANDOFF.md` (23 fresh GPU gates,
> 40293 exact metrics; no arena03/model PASS inheritance).
> It is not a new public release/admission: a freshly reviewed serving build and
> matched-4GiB A/B are still required. Arena snapshots use the explicit v2 root
> partition; do not apply the old segmented-component sum to them.


Status: **final v2 release CPU-audited; fresh smoke metadata prepared; GPU authorization pending**.

Final build: `target/validation/tp2-mixed-decoded-cache-public-release-20261010-03`.
Handoff SHA256: `90a8289e18334aaae4dde93f0ae763868faf3b78e666258fddedd96e6e7b09fd`.
Binary SHA256: `dadf831eb2af477108efd402171c40f7544a2600d14b8e5e7ab8a456d2625e5a`.
The import-only rebuild is byte-identical to -02, but an independently rehashed
new inode/build with separate receipts. No old handoff was rewritten.

CPU-only artifacts now available:

- `admission/decoded-cache-20261010-03/smoke-admission.json`
- `admission/decoded-cache-20261010-03/model-runtime-quality.json`
- `runs/tp2-decoded-cache-smoke-20261010-001/prepared.json` (S1/S2 only)

There is no new serving-smoke receipt, accepted pair, profile, or GPU result.
Pair/profile promotion remains blocked until fresh same-binary S1/S2 succeeds.
No new smoke, pair, trace, GPU query, lock acquisition, or server launch was
performed in this preparation. No production source, tool installation,
historical receipt, or deleted harness was changed/restored.

## Ready paper entry points

From the repository root, this command only inspects CPU-readable sources and
prints a plan; it does not prepare an accepted run or synthesize a binary hash:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/tp2_resident/paired.py plan \
  --purpose paired --serving-scope decoded-cache-4g
```

`--serving-scope` is plan-only. Prepare/run obtain the exact scope from the
hashed admission. Ferrule engine metadata must carry:

```json
"serving_scope": {
  "policy": "decoded-cache-workspace-v1",
  "cuda_workspace_mb": 4096,
  "cuda_device_budget_mb": 22528
}
```

`paired.ferrule_spec(binary, serving_scope)` changes only
`--cuda-workspace-mb 1024` to `4096`. Every other argument and the isolated
engine environment remain unchanged. The default/old policy still uses 1024.
The new prepared manifest records `serving_scope_difference`; it does **not**
assert that changing workspace enables cache. The runtime release must prove
that separately. Unknown scope values are rejected, not arbitrary flag overrides.

The intended decoded cache is **3 GiB total per owner**, including 1340 MiB
fixed pins plus routed expert LRU/physical pool, not 3 GiB plus fixed pins.
Workspace also includes operation grants. A 4096 MiB ceiling is a proposed
admission ceiling, not proof of the actual derived requirement or memory peak.
Keep each total device ceiling at 22528 MiB (22 GiB); recheck complete production
resource admission, including context/allocator margins, after the build.
Do not increase the device cap to compensate for a workspace rejection.

The workload stays PP1 TP2 EP1, thread ranks, BF16 activation/KV/conv, F32 SSM,
resident FP8 checkpoint shards, no CPU offload, ctx128, batch8, seq2,
concurrency1, QA8. Preserve the existing pinned vLLM Python/cu129 runtime,
TP2/EP-disabled configuration, NCCL SHM environment, Marlin and UUID order.
`fairmatch=false`, `exact_speedup=null` remain mandatory. The old
`compare_pilots.same_scope` intentionally rejects the workspace flag change;
do not weaken it or relabel a changed-scope descriptive comparison as identical.

## Typed release and source audit now connected

`optimized_release.audit/bootstrap/validate` dispatches the actual cache release
v1/v2 schemas to `decoded_cache_release.py`, a CPU-only receipt adapter, not a
launcher. The historical 19-gate optimized schema is retained separately.
Cache quality is authenticated against the concrete `decoded_cache_gate.ALL_GATES`
selectors: six small operator/cache gates, full40, and the six named lifecycle
recipes. The frozen summary SHA, test-binary pins, authenticated supervisor and
worker launch IDs, ECHILD, idle/release, exact comparator paths/reference hashes,
canonical manifests, metric counts and owner cache bindings are checked.
The result is **13 gates / 37 calls / 40293 exact metrics**, not public GPU PASS.

`decoded-cache-source-review.json` enumerates all **17 changed source inputs**
(including three CPU test inputs), **two new production snapshot/logging files**,
and all **13 added dependency inputs** (including compile docs and CUTLASS
version metadata). Unlisted drift rejects. The public policy/program-plumbing
changes remain distinct from the old GPU-tested source map.

Three changed implementation files have stronger evidence: removing explicitly
reviewed observation additions from `mixed_decoded.rs`, `hybrid_rank/mixed.rs`
and `rank_performance.rs` reconstructs the original quality SHA byte-for-byte.
The unchanged backend/operator source set is independently hash-checked. This
supports the unchanged cache/operator relationship; it does **not** transfer
GPU/public PASS to the source-different release. The additional unused import
removal independently reconstructs the -02 builder source hash.

The v2 build audit checks real source stability, dependencies, release flags,
CUDA12.4/sm_86 environment, runtime library pins, independently copied binary,
strict Clippy commands and exact CPU selectors. It records **15 prior CPU tests
(3 ignored), reused**, separately from **3 freshly rerun snapshot tests** and
public compile/strict lint checks. Prior tests are not called a fresh rerun.
The old -02 metadata-only cache ledger remains labelled as prior CPU evidence:
per owner workspace 3,875,939,840 bytes under 4 GiB and total 23,041,273,856 bytes
under 22 GiB. Its fixture device ordinals 3/1 are not the live UUID assignment.

Cache metadata explicitly binds enabled=true, 3 GiB capacity, 1340 MiB fixed
pin target, 768 MiB routed working-set grants and no inherited numerical PASS.
The existing lifecycle assessor now requires both owners' **ready and shutdown**
snapshots. It validates component sums and current/peak physical totals <=3 GiB,
publication, stable image identity, idle/no quarantine and no pending leases.
Missing, duplicate, foreign, `shutdown_unproven` or over-budget rows reject.
Snapshot bytes remain cache-only host ledgers, never global NVML or CUDA fence
proof. Normal shutdown and authenticated drain/release are independently required.

No-trace smoke/pair requires rankperf disabled and `profiled_activity=null`.
Profile requires rankperf enabled, null activity at readiness and actual collected
activity at shutdown. Hit fractions are derived only when counts were collected;
null is never converted to zero. Shutdown activity is cumulative across enabled
invocations, **not** a measured-request-only kernel count.

The actual subscriber reads **FERRULE_LOG**, not RUST_LOG. In the cache-on profile
slot only, the existing adapter sets
`info,ferrule_request=debug,ferrule_performance=debug`; the pinned runtime calls
`set_mixed_rank_performance(tracing::enabled!(...))` and the model checks the
second logging gate. Ordinary paired `FERRULE_LOG=info` stays untouched; smoke
adds request debug only. Profile logging overhead must be disclosed and its
timings never combined with the no-trace benchmark.

CPU bootstrap is now executable through `optimized_release.py bootstrap` with
`--handoff`, explicit final `--handoff-sha256`, and a fresh `--output`. It rejects
missing/mismatched hashes before staging. `audit` validates without staging.
No artifact/prepared receipt is overwritten or resumed.

## Existing lifecycle sequence after handoff and fresh authorization

1. Build a fresh smoke admission through the reviewed release audit, then use
   `paired.py prepare --purpose serving_smoke` and the existing separately
   authorized `run`. S1/S2 must qualify the **new binary and new exact spec**.
2. Promote only that same-binary smoke to fresh pair/profile admissions.
   `optimized_release.promote` reuses the same downstream flow and automatically
   selects the current-release official-tool policy for the cache-on profile. Run the no-trace pair in
   the unchanged order **F1,V1 / V2,F2**. Preserve excluded first request,
   two warmups, five measured QA8 requests per slot. Keep rank performance,
   numerical trace, and profiler off in these benchmark blocks.
3. Use the same qualified binary/spec for one short official profile through
   `paired.py`, with a fresh run-bound eight-UUID instrumentation grant and
   sole-owner authorization. Collector B0/B1 accounting, UUID locks,
   authenticated supervisor/subreaper, ECHILD, drain, release, fail-stop and
   no-resume rules are unchanged. A document or old grant authorizes nothing.

For the **new** profile admission, set the top-level `profiler` to
`profile_2025.TOOL_CONFIG`:

```json
"profiler": {
  "release": "2025.3.1.90-253135822126v0",
  "policy": "current_release_same_binary_smoke"
}
```

This selects the already installed official CLI/importer/CUPTI, verifies their
three established SHA256 pins and retains all current-source/build/quality/
same-binary-smoke checks. It is rejected for smoke or no-trace paired purposes.
`profile_2025.CONFIG` remains solely the historical frozen-4d40 policy; its
fixed audit is not attached to a new-release preparation. No tool is installed
or executed by selecting/validating the policy on CPU.

The short profile remains Hello/canaries followed by explicit start-later,
**one fixed QA8 warmup plus one measured QA8 request**, then stop. Both request
windows and the complete export are audited. Do not change workload or trim
away a bad warmup to obtain acceptance. Request debug logging is retained for
identity/window markers. Rank-performance logging requires the two real
switches above; enabling the logging target alone is insufficient.

## Answering whether decode_weights / pack really decreased

Reuse `profile_trace_audit.py` (v2 correlation implementation), with a fresh
output name under the new run after official export and authenticated release.
Do not regenerate or overwrite the old raw/export/audit/launch receipts.

- Bind each window using the HTTP/SSE request ID, server admission/terminal
  markers and official session epoch; correlate process/thread/device UUIDs.
- Report **measured-request** kernel count and summed duration for
  `decode_weights` and `pack_bf16_activation`, separately by owner when possible;
  also retain CUDA API and DMA counts/durations/bytes and warmup quantities.
- Keep both correlation directions: activity to unique successful API and
  successful launch/copy/memset API to activity, including full-export search,
  cross-window checks and same-thread overlap checks. Keep importer warnings.
- The old official request6 observed reference is 26592 decode_weights kernels /
  270507218 ns and 19228 activation-pack kernels / 30012786 ns. Its
  `observed_correlation_closed` is not full-trace acceptance:
  **`full_trace_accepted=false`**, with warmup missing records and a verifier
  warning. New counts are comparable local observations only after the same
  audit and request/load check; paired record loss cannot be ruled out.
- Cache reduces weight materialization in principle, but activation pack may
  remain. Do not declare either reduced before observing the new measurements.
  Summed kernel/API durations overlap across owners and are not wall time.
- Exact prefill/decode attribution requires real phase markers and a verified
  mapping to the official trace. First-emission splits remain explicit proxies;
  neither kernel names nor a guessed token boundary establish exact phases.
- Keep no-trace throughput/TTFT/startup separate from profile durations. A
  changed workspace ceiling precludes identical-flags cache-only attribution.

## CPU validation in this preparation

- Baseline: 193 TP2 paper unit tests passed.
- Initial workspace/tool preparation: 200 tests passed.
- Release-adapter follow-up: **215 tests passed**, including the real final v2
  handoff audit, 13-gate evidence, observation-removal proofs, source drift,
  cache metadata, final hash requirement, profile-only logging and snapshot
  cap/null/identity/fail-closed checks.
- Official CLI/importer/CUPTI files were SHA256-read only; all three matched
  `OFFICIAL-NVIDIA-TOOLS.md`. No profiler execution or fresh GPU query occurred.

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover \
  -s papers/arxiv/experiments/tp2_resident -p 'test_*.py'
```
