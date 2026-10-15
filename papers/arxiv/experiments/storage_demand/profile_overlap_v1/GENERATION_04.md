# Generation -04: owner-safe wrapper, model-scoped overlap

**CPU-prepared; no capture authorization. Request fresh PARENT GO only after
pressure cleanup and the parent's Git phase.** The pressure agent retains GPU
exclusivity until the parent releases it. Preparation/check/tests never query a
GPU, invoke `nsys`, acquire a GPU lock, signal a process, or launch the model.

This is a new plan, not a continuation or reinterpretation of old evidence:

| Private generation | Preserved disposition |
| --- | --- |
| `artifacts-go-20261015-01` | Rejected; no upgraded acceptance |
| `artifacts-go-20261015-02` | Blocked; no capture |
| `artifacts-go-20261015-03` | Blocked direct-target proposal; no capture |
| `artifacts-go-20261015-04` | Fresh source/binding seals; awaits fresh parent GO |

All predecessor files are hashed in the new private preservation receipt and
checked again without rewriting them. Old GO receipts cannot authorize -04.
The ordinary paired raw data, production source/binary, and manuscript are not
modified. `SCOPE_20261015.md` remains the historical -03 decision record.

## Lifecycle decision

Use the **original maintained collector/control path**:

```text
authenticated worker -> nsys -> traced Python target -> frozen CLI
```

The traced Python target invokes the original `_profile_capture.run_target`.
Its `Popen` still owns the unreaped CLI child. Only that wrapper performs the
maintained direct-child termination/wait. The outer worker writes `profile-stop`,
waits for the model/B1 handshake, and lets the collector exit naturally; the
maintained authenticated subreaper owns descendant drain on failures.

No new execution loop, direct-CLI-under-nsys target, numeric grandchild PID
signal, process-group signal, or `nsys shutdown --kill` is used. The recorded
self-only `pidfd_open` ENOSYS result is preserved as -03 evidence, not reprobed or
worked around. CPU tests check private capture/live/scope aliases and function
globals without mutating the maintained TP2 modules.

One pair only: **blocking, then lookahead**, exactly one frozen `canary-qa`
request generating eight tokens per arm. No Hello, warmup, pilot, retry,
replacement, or extra generation request. GET-models readiness is not inference.
Startup is bounded at 120 seconds from collector launch; capture/wrapper lifetime
150 seconds; direct-child shutdown 45 seconds; authenticated worker supervisor
300 seconds. The single request has at most 60 seconds and must finish before
the original capture cutoff. Cleanup/export retains the existing bounded
handshake. No deadlines slide after a slow startup or failed permission check.

## Predeclared collection and source contract

- `--trace=cuda,nvtx,osrt --sample=none --cpuctxsw=process-tree`.
- GPU metrics and GPU context switches off; file-access OSRT on; threshold
  1000 ns (official help says file APIs bypass this threshold).
- `--duration=150 --kill=none --wait=all --stop-on-exit=true`; full non-lazy
  SQLite export, no filters or time shifts.
- Reuse the sealed official 2025.3.1.90 help and CPU-environment check from -03;
  verify the installed tool hash, boot ID and current perf paranoid setting.
  **No fresh tool invocation occurs in this CPU task.** Reused permission
  evidence is not a guarantee that later collection succeeds. Actual permission
  rejection, absent scheduling under requested process-tree, or any other
  unexpected warning stops the arm and leaves acceptance false. Never remove a
  flag or warning after inspecting data.
- Preserve all 19 explicitly verified runtime ELFs, including all 12 original
  CLI requirements and the 16 required profile libraries. Optional means checked
  if mapped, not an exemption for arbitrary paths. Deleted executable/library
  mappings still fail; no global deletion suppression is added. Pathname/inode
  evidence is not live mapped-memory hashing.
- The immutable CLI SHA remains
  `7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01`.
  Archive-bound source checks verify the global CLI subscriber, per-submission
  worker dispatcher, current NVTX numeric-field layer, read/upload/math spans,
  and typed decoder read counters. `FERRULE_PROFILE_NVTX=1` is profile-only.
- Keep the existing JSON logging filter: performance statistics at DEBUG,
  exact token IDs at TRACE. Per-read Debug-formatted receipts are TRACE-only and
  are **not** enabled or parsed as a producer-identity journal. The existing
  filename trace (`FERRULE_IO_TRACE`) remains absent. Model routing, image,
  quantization, source and CLI arguments are unchanged.

## Role binding and acceptance

At Ready and after the only QA, persist private read-only live observations of
both the model and traced Python wrapper: PID/birth/parent, live state before and
after, actual executable SHA, exact argv, source/prepared SHA, authenticated
manifest/launch/gate and arm/generation. The model must be the wrapper's direct
child. The collector is control/custody evidence, **not** a warning-exempt role.
Offline analysis joins these observations uniquely to raw SQL process keys,
retaining VM bits. A numeric PID list, plausible process name, arbitrary hash,
reused PID, stale generation, or ambiguous raw diagnostic PID is insufficient.

`scoped_sql_evidence_consistent` admits the next arm, but is not final acceptance.
Only final independent raw checks can produce `scoped_overlap_accepted=true`:

1. Exact HTTP/token/terminal QA join and request clocks.
2. Actual model and wrapper PID/birth/executable/source binding, plus both raw
   map snapshots using the unchanged role-aware validator.
3. Whole-export bidirectional process-key/API/kernel closure, no orphans,
   duplicates, unknown graph launches, failed launches, or foreign payload.
   Only actual kernels correlated to expert math submission ranges enter the
   compute union; startup kernels, copies, uploads and waits do not.
4. Positive complete numeric read/math/upload NVTX ranges; worker OSRT
   `pread`/`pread64` intervals and bytes reconcile to each read range, current
   typed decoder deltas, final balanced read counts, and actual owner image
   generation. Last receipt provides one source/operation check, not a complete
   operation/layer/expert/producer journal. No expected native kernel count is
   invented from model shape or facade counters.
5. No model/universal/unknown-PID warning, unknown diagnostic severity, or
   reported collector loss in **any** process or severity. Only exact official
   no-NVTX/no-CUDA/CUDA-not-started diagnostics for the independently bound,
   actually payload-free Python wrapper are outside the model overlap scope.
   This is role classification, not a blanket text whitelist.
6. Dynamic owner closure, successful collector/full export, authenticated ECHILD,
   and original all-eight idle, including maintained `gpu_idle_window.v2`
   post-ECHILD observations. Offline acceptance replays the raw V2 samples;
   saved PASS flags alone are insufficient. These facts are not a universal
   physical-release certificate.

Every official warning still rejects **full-trace** acceptance. Thus scoped
acceptance can pass with bound wrapper-only warnings while
`full_trace_accepted=false`. Unknown dropped-event counts remain **null**, never
zero. Simultaneous unreported API/activity loss is not excluded. No historical
or diagnostic-partial report is eligible for an upgrade.

The claim is intersection of interval unions: actual expert GPU kernels with
(a) worker allocation-plus-read envelopes and (b) userspace pread wall intervals,
reported separately. Pread wall time includes preemption and possible page-cache
service; it is neither active IO nor NAS backend time. Thread-state analysis is
not required for this mathematical overlap criterion and remains unestablished.
This cross-process profiled Python-wrapper run includes startup, shutdown and
collector overhead. It is **not ordinary performance**, a full-memory bound,
or proof of universal trace completeness. Producer-level identity is partial.

## CPU entry and conditional parent GO

From the repository root, CPU-only:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/profile_overlap_v1 -p 'test_*.py' -v
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/profile_overlap_v1/run.py plan
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/profile_overlap_v1/run.py check --directory papers/arxiv/experiments/storage_demand/profile_overlap_v1/artifacts-go-20261015-04
```

The fresh preparation is exclusive-create. Do not rerun `prepare` over an
existing seal or edit source after sealing; another change requires a new
parent-reviewed generation, not rewriting -04. No GO is generated here.

After pressure cleanup, request **PARENT GO for the sealed -04 SHA**. The fresh
immutable receipt must contain the original README's schema/directory/prepared
SHA/order/pair ceiling/sole-owner/all-eight/ordinary-closeout/boot fields, plus:

- `issued_unix_ns` and `expires_unix_ns`, current and at most 900 seconds apart;
- `pressure_cleanup_complete: true` and `pressure_closeout: {path, sha256}` for
  the actual parent's completed pressure cleanup evidence;
- `selected_topology: "maintained_traced_python_wrapper_owns_unreaped_cli"`;
- `warning_decision` exactly equal to the prepared `overlap_scope`.

The parent supplies that real receipt path and hash to `run.py run` using
`--directory`, `--parent-go`, `--parent-go-sha256` and
`--sole-gpu-owner-granted`. Do not invoke it until the parent grants GO. Each arm
is audited before the next is admitted; a failure permanently stops the attempt.
After both arms and cleanup, `run.py accept --directory` is CPU-only and computes
scoped and full-trace results separately.

## Exact staging whitelist (after the parent's Git phase)

No Git command is part of this CPU task. The following paths are relative to
`papers/arxiv/experiments/storage_demand/profile_overlap_v1/` and are the complete
source/test/docs whitelist; no recursive directory add is safe:

```text
README.md
RUN_20261015.md
SCOPE_20261015.md
GENERATION_04.md
analyze.py
protocol.py
role_context.py
run.py
runtime_contract.py
scope_contract.py
tp1_collector.py
fixture.sql
test_collector_contract.py
test_profile_overlap.py
test_role_context.py
test_scoped_contract.py
```

`RUN_20261015.md`, `SCOPE_20261015.md` and `fixture.sql` are retained unchanged.
Exclude every `artifacts-*` directory, GO/attempt marker, private log/raw map,
SQLite/report, ordinary paired file, production file and manuscript. `/papers/`
is ignored; only the parent should deliberately stage these exact source paths
when its pressure/Git phase is complete. Do not commit private preparation seals.
