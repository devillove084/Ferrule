# Additive TP1 actual-pilot execution adapter

`run_pilot.py` is the new entry. It reuses the maintained `contract`,
`launch_guard.seal/authenticate`, `run_process` Linux subreaper/ECHILD,
`_profile_capture.run_target`, HTTP client, and observer. It is not a second
supervisor or general experiment harness. Original `protocol.json`, case,
prepared/validation receipts, historical middle runner, results and analyses
remain unchanged. The original README describes the original preparation-only
entry; this document describes the additive execution entry.

**Current state: candidate unbound; CPU validation only.** This task performs
no GPU query, GPU lock acquisition, GPU launch, numerical acceptance, or actual
pilot. Mock traces/results are fixtures, not experiment receipts. Do not infer
permission to execute from a CPU seal. Instrumented token IDs/raw SSE can
reconstruct output; raw mountstats may expose exports. Keep artifacts in approved
local custody, do not upload them, and do not dump arbitrary environment/secrets.

## Fixed execution contract

- Exactly `L1,L2,L3,L4,L5`, no retries/replacement trials. Exactly the sealed
  12 named requests in each: four initial, one excluded-first QA, two warmups,
  five measured QA. All 60 requests and 25 measured slots remain scheduled
  after any STOP. HTTP success and quality/metric eligibility are separate.
  Pilot-level eligible measured numerators additionally require complete owner
  close, normal exit, authenticated drain and post-idle; request-only eligibility
  is retained separately and cannot leak through a failed lifecycle assessment.
- Startup 700s from launch; request <=120s; aggregate requests <=480s without
  sliding; target inclusive lifetime <=1240s; direct-child shutdown <=45s;
  supervisor application timeout 1340s; outer lifecycle 1440s; pilot 7200s.
  Each admitted request journals its absolute deadline before HTTP. Lost
  issued outcomes are censored, never relabeled unissued. No traffic is sent
  to compensate failure. Idle/query/validation/log costs consume the existing
  reserves; insufficient reserve stops rather than extends admission.
- The renewed sole-owner grant, exact ordered eight physical UUIDs, selected
  UUID, topology, candidate, argv/env, schedule, limits and helper pins are
  sealed into each worker contract. Selected UUID is exposed as CUDA ordinal0;
  it is never guessed from an ordinal. `contract.exclusive_owner_lock` is a
  nonblocking exact-UUID **local** lock, not a global/NFS reservation.
- Pre-idle uses strict `scripts/gpu_idle_window_v2.py` all-eight zero memory,
  utilization and compute-app checks. Post-idle may use that version's bounded
  utilization-only stabilization **only after authenticated ECHILD and absent
  recorded PID/birth owners**. Unknown custody does not admit stabilization or
  another lifecycle. Historical idle helpers/receipts are not upgraded.
- Maintained observer binds the real capture-child/server-grandchild ancestry,
  selected capacity, actual UUID visibility, sampled GPU/PSS and original
  PID/birth/parent. The capture wrapper owns server Popen/TERM; the adapter
  requests shutdown with `profile-stop`. Neither observer nor coordinator
  signals server/foreign numeric PIDs or groups. The outer maintained custodian
  retains sole descendant cleanup authority. Known foreign compute apps,
  nonselected positive memory delta >1MiB, lost identities, pending/unknown
  physical closure, trace/counter/reset/quality failures all stop progression.
- Source/build/quality qualification is fully revalidated before every
  lifecycle and in its authenticated worker. Bound receipt/artifact/helper
  pins and UUID lock are checked between serialized requests. No repeated
  heavyweight build inventory scan occurs inside a measured HTTP call.

Deadlines close admission, not authenticated cleanup authority. An uninterruptible
OS operation cannot be promised to complete physically by a wall deadline.
Overruns or unverifiable cleanup are retained as failures and prohibit the next
lifecycle; process exit, Drop or apparent idle never repair Unknown.

## Evidence and receipt ownership

Every final adapter receipt uses `contract.write` exclusive creation. New seals
use new names; a run requires a new `artifacts-NAME` directory directly under
this package. No overwrite/resume/retry operation exists. The maintained
subreaper custody journal, HTTP checkpoint and capture status files retain
**their existing mutable checkpoint contracts**; the adapter never upgrades
those intermediate files to append-only immutable evidence. It archives and
hashes the drained artifacts before final assessment.

Per request, `issued.json` precedes maintained HTTP dispatch; `http.json` and
`http-terminal.json` retain actual SSE/usage/terminal evidence; `observed.json`
retains actual `cmpl-ferrule-N` joins, successful enqueue ordinals and generated
IDs, worker-terminal enqueue result/usage, HTTP-side terminal, log byte offsets,
logical counters, process-I/O and NFS pre/post windows. No retokenization or
request-name/ordinal-to-server-ID guessing. Closed-channel/backpressure or
terminal usage mismatches suppress eligibility even when HTTP succeeded.
Final whole-log reconciliation occurs after normal capture shutdown/log drain.

`resource_shutdown_summary.summary_json` is parsed with actual owner/generation
fields and qualifiers. Its actual first-attempt scopes are mapped to the single
owned server log, with original PID/birth/parent and physical UUID retained.
This is **not** a separately observed Ready resource-scope snapshot; scopes are
not PIDs. Changed scopes or any explicit Unknown attempt remain sticky. Missing
summary/scope/quiescence stays unverified and blocks continuation. A complete
closed/quiescent retained owner can proceed only with normal exit, authenticated
drain and independent post-idle. `resource_contract_complete=unverified` and
`universal_final_release_verified=false` remain true policy qualifiers; no
post-Drop/universal resource-release certificate is manufactured.

Raw mountstats and parsed samples bracket prelaunch/Ready, entire request phase,
every request and shutdown. The authenticated worker is the persistent reader;
its boot/mount namespace is sealed to the live server, and all checkpoint shard
paths must fit the same longest NFS mount. Identity/age/counter reset or a missing
sample suppresses all deltas and stops. Shared-client `serverreadbytes`/READ RPC
counters remain unattributable to this server. Physical NAS backend disk/network
bytes remain unknown. Authenticated `/proc/PID/io` counters are separately labeled
Linux PID accounting, **not** NFS physical traffic or backend NAS I/O.

Actual loaded shared-library mappings at Ready and after requests must be
covered by the owner-supplied runtime pin manifest and unchanged. No dynamic
library binding is inferred from installed inventory alone. Observations are
instrumented, descriptive and sampled, not exact peaks, a host cap, storage-cold
proof, throughput speedup, statistical significance or scientific PASS.

## CPU-only commands (repository root)

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/tests -p test_execution_v2.py -v
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/tests -v
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/newversionedprotocol/protocol.py check
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/newversionedprotocol/run_pilot.py plan
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/newversionedprotocol/run_pilot.py check
```

Authoring only: `seal-execution --output` must name a **new** seal in this
package; never rerun it against an existing file. The default read-only seal is
`execution-prepared-20261014-01.json`. Later code/helper/doc/test changes require
a systematically new seal and an explicitly selected `--execution-prepared`;
no automatic resealing is offered. The original `protocol.py` still has no run
mode; candidate qualification/binding remains its existing CPU-readable gate.

## Prerequisites before the new `run` mode

1. A **new current** CUDA-only opt3/fat-LTO/codegen1 CLI artifact and maintained
   build receipt meeting `new-baseline-build-requirements.json`; immutable staged
   binary, source archive/manifest, native/compiler/runtime inventories,
   GPU-hidden exact-binary serve-help receipt and actual instrumentation sinks.
2. Passing actual current StorageMiddle64/maxdecode2 full40 logits/routes/states,
   retained fresh-generation and cold-owner-restart reports, independent reference
   manifests, authenticated producer custody/test executables, and a verified
   compiled production-source relation to that exact CLI build. Historical -02
   binary or earlier numerical PASS is not current acceptance.
3. Owner supplies the actual candidate-binding input and uses the existing
   `protocol.py bind-candidate` gate to produce a new final-candidate seal.
   `run_pilot.py` only validates it; it does not synthesize candidate bindings,
   build artifacts, quality reports, or producer identity.
4. Reviewed current execution seal, pinned runtime mapping coverage, readable
   passive NFS mountstats, renewed explicit sole-owner consent, literal selected
   physical UUID plus exact ordered eight-UUID inventory, and a fresh local output.

The new run entry is `run_pilot.py run` with required `--candidate`,
`--execution-prepared`, `--output`, `--gpu-uuid`, `--eight-uuids` (eight literal
values), `--sole-gpu-owner-granted` and `--owner-note`. No executable GPU command
is supplied until real artifact paths and owner authorization are available.
No-GPU end-to-end tests mock qualification and hardware boundaries, exercise
actual maintained HTTP `main`/Stream and actual `run_target` with mocked
`Popen`/`DEVNULL`, and never create an acceptable candidate seal.

## Explicit CPU builder adapter and revised workflow registration

See [BUILD_RECEIPTS_V1.md](BUILD_RECEIPTS_V1.md) for the reviewed
`ferrule.cuda-only-cli-release-build.v1` adapter, exact descriptor/API fields,
CPU build versus GPU producer/run coverage, strict same-cut all-three qualification,
and owner-requested **new-name** `seal-workflow`/`--cpu-prepared` selection.
The original requirements JSON describes the unchanged legacy M7 builder; it is
not rewritten to impersonate this independent CPU builder receipt.

Existing preparation/execution seals intentionally fail `check` after these
helper changes. Planning exposes that blocker without authorizing or writing a
seal. No stale pin is ignored by candidate binding, execution check or run.
The current `a9fc…` CLI remains quality-pending and unbound.
[PAIRED_READ_PROTOCOL_DRAFT_V1.md](PAIRED_READ_PROTOCOL_DRAFT_V1.md) is a separate
future-version design draft, not an amendment to the fixed five-cycle baseline.
