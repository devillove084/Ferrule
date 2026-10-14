# Paired expert-read study v1 — CPU preregistration

**No candidate, GPU PASS, server outcome, NAS measurement or authorization is
created here.** The production source cut is `033519c3def098ce417e4b6ff0f141a9f30bddb1`.
The baseline `newversionedprotocol` five-lifecycle / 60-request / 25-measured-QA
protocol, cases, prepared seals and historical results remain unchanged.

## Frozen design and estimands

`protocol.json`, `cases.json`, `requests.json`, `oracles.json`, `order.json` and
`source-cut.json` are frozen before outcomes. `cpu-sealed.json` binds the design,
source inventory and maintained workflow. CPU checks do not hash/read the NAS
model/tokenizer paths; their historical reference hashes are recorded, not
re-certified here. Actual owner-side candidate binding must verify independent
reference manifests and checkpoint/tokenizer identity before serving.

Five independent paired **blocks**, each with two fresh server lifecycles:

| Block | First | Second |
|---|---|---|
| P1 | Blocking | Lookahead |
| P2 | Blocking | Lookahead |
| P3 | Lookahead | Blocking |
| P4 | Lookahead | Blocking |
| P5 | Blocking | Lookahead |

Seed `20261015`; SHA256 of `20261015:P1` through `20261015:P5` ranks the blocks;
the first three ranked blocks are Blocking-first. Execute blocks in P1..P5
order. This is deterministic counterbalancing, not outcome-dependent allocation.
The unavoidable first-arm imbalance is 3:2. Within each block both arms must use
**the same immutable staged CLI binary**, physical UUID, checkpoint/reference
pins, environment and budgets; the sole argv difference is the value of
`--expert-read-schedule` (`blocking` / `lookahead`). Enabled Blocking is the
control; the disabled historical pipeline is not an arm.

Both arms use `--expert-read-host-mb 64 --expert-read-metadata-mb 8
--expert-read-worker-stack-mb 2`: 64 MiB includes 8 MiB metadata and a 2 MiB
requested stack, leaving 54 MiB payload reservation (27 MiB packed + 27 MiB
artifact). OS stack rounding, TLS and allocator overhead are not included and
require host headroom. CUDA cache: 64 entries, 335,544,320 bytes (320 MiB,
**including** 8 MiB scratch); device 7,516,192,768 bytes (7 GiB). Workspace 2 GiB,
KV 16 MiB, context 128, PP/TP/EP=1; no 8k, topology sweep or extra request families.

Each lifecycle retains the existing 12-request serial schedule: four initial
canaries, one first-QA excluded, two QA warmups, five measured QA. Ordinary totals
are **10 lifecycles, 120 scheduled requests, 50 scheduled measured QA**. The QA
input is exactly `The capital of Japan is`, IDs `[760,6511,314,6124,369]`, raw
completion (no chat template/special tokens), temperature 0, seed 1, generation
8. Exact output text: ` Tokyo.\nThe capital of Japan is`; exact generated IDs:
`[25358,13,198,760,6511,314,6124,369]`. Actual enqueue token IDs, terminal usage,
HTTP request IDs and SSE/DONE are reconciled; worker-channel enqueue is **not**
HTTP delivery or GPU completion. All three generation8 canary oracles are frozen.

Primary statistic: compute each arm's median of its five measured-QA client E2E
latencies, then each block's `log(Lookahead/Blocking)`; report the exponentiated
median of the five block log ratios. Prespecified secondary metrics use the same
pairing for client TTFT (first nonempty SSE content) and quality-qualified goodput:
40 actual generated tokens divided by the sum of the five client E2E durations
for a complete qualified arm. This is serial request-window goodput, **not**
wall-clock startup throughput, sustained concurrency or general serving throughput.

`analysis.py` enumerates all `5^5=3125` paired-block bootstrap draws and uses
nearest-rank 2.5%/97.5% endpoints (ranks 79 and 3047). Resampling unit is the block,
never one of the 50 repeated requests. All five pairs must qualify for an aggregate
or interval; otherwise all five block rows remain and the aggregate is null.
The interval is exploratory/descriptive n=5, not a significance test, power
calculation, guaranteed-coverage CI or SOTA claim. Failed pairs are never dropped
or adaptively replaced. Separate servers do not prove independent page-cache/NFS
samples or cold storage.

## Runtime envelope and custody

Absolute, non-sliding monotonic **7,200 s** from coordinator registration, with
**100 s study finalization/cleanup reserve**. Each next lifecycle requires a full
650 s plus the reserve before admission. Ten worst-case envelopes total 6,500 s;
6,600 s including final reserve fits two hours. No deadline is reset to compensate
for delay or failure.

| Phase | Maximum seconds |
|---|---:|
| Coordinator pre-idle | 20 |
| Worker pre-idle | 20 |
| Startup | 120 |
| One request | 60 |
| All 12 requests, aggregate | 300 |
| Worker bookkeeping | 40 |
| Target lifetime (120+300+40) | 460 |
| Target shutdown | 30 |
| Worker finalization | 20 |
| Maintained supervisor (20+460+30+20) | 530 |
| Supervisor handshake/cancel/reap call overhead | 28 |
| Coordinator post-idle | 20 |
| Coordinator finalization | 32 |
| Outer lifecycle | 650 |

The historical n=1 QA observation was 18.309471419081092 s E2E and startup
18.497 s; 12 QA-like requests extrapolate to ~220 s per lifecycle, or ~37 minutes
HTTP cost for ten. A planning allowance of 240 s per lifecycle adds 40 minutes,
~77 minutes total. These are **feasibility extrapolations, not current-candidate
measurements**. If the treatment/startup is slower, the finite deadlines censor
and STOP; no budget enlargement or replacement. The aggregate 300 s does not
promise `12*60` seconds.

`run_paired.py` specializes only the maintained TP1 worker's entry/directory and
sealed contract in a process-local context manager. It reuses:

- `newversionedprotocol/run_pilot.py:worker,supervise,idle,baseline,PilotMonitor`
- `runnerhelpers/execution_v2.py:serial_requests,launch_capture,capture_root,
  library_maps,owner_summary,lifecycle_assessment,recover_requests`
- existing `launch_guard`, `_profile_capture.run_target`, authenticated
  `test_vllm_qwen35.run_process` subreaper, passive NFS sampler and observer.

No replacement HTTP/process/signal harness is added. Qualification is checked
before output reservation, locking or hardware query, again before each lifecycle
and inside the authenticated worker. Cancellation or a gate failure closes
admission but retains maintained cleanup authority. Unknown/pending owner,
quarantine, changed identity/pins, failed trace/token/text/usage/counter/NFS gate,
failed post-idle or timeout STOPs all subsequent slots. Unfinished admitted
requests remain censored; remaining planned slots remain unissued. Only a normal
owner-local quiescent close, authenticated ECHILD drain and independent all-eight
post-idle permit progression. Final read-pipeline ledger must prove the selected
enabled mode, positive reads and bounded known charge; exit/idle cannot repair
unknown custody. Retained owner-local residency is not invented as zero or
universal final release. Samples are 2 s; nonselected delta tolerance 1 MiB.

## Actual candidate gate (currently blocked)

`protocol.py bind-candidate` accepts only a new owner-produced input descriptor
with schema `ferrule.storage-demand.paired-read-candidate-input.v1`, status
`quality_passed_cpu_bind_requested`, actual `protocol_sha256` / `cases_sha256`,
`artifact_pins` and the following evidence. There is **no example candidate blob**.

1. One immutable fresh CLI artifact; actual Cargo/source archive, help/version,
   native archives, full compiled dependency/compiler/CUDA/CUTLASS/runtime pins.
   `build_adapter.py` is the isolated explicitly selected
   `ferrule.storage-demand.paired-nvtx-cli-build-adapter.v1`; raw build schema
   `ferrule.paired-read-nvtx-cli-release-build.v1`. Its reviewed feature list is
   exactly `cuda,nvtx-profile`, never CPU-reference. It reuses the maintained
   archive/source and help/ldd structural reviewers; original v1 remains unchanged.
   `FERRULE_NVTX_INCLUDE_DIR` is an explicit build allowlist key and the entire
   official `nvtx3` header tree must be compiler-input pinned. No fabricated M7
   receipt, schema guessing or feature normalization is accepted.
2. Actual same-cut constructor report (all four BF16/W8A16 x Blocking/Lookahead
   cells), then each arm's full40, retained diagnostic fresh, ordinary
   no-diagnostic-observer equality, and cold-owner reports. Numerical tolerances
   are `atol=0.00025`, `rtol=0.012`; output IDs/text are exact. Ordinary fresh
   fingerprints must match the pinned diagnostic report's 64 actual logit/greedy
   call fingerprints exactly. Both endpoint read ledgers must be known/bounded;
   configured-but-never-read does not qualify.
3. Each `quality[mode][label]` record (`full40`, `fresh`, `ordinary`, `cold`) and
   the `constructor` record names pinned absolute `report`, `executable`,
   `supervisor`, `authenticated_launch`, `source_manifest`, `producer_final` paths;
   diagnostic gates additionally name `reference_manifest`. Source manifests
   bind the actually compiled CLI and quality executable to the frozen source cut.
   Existing full40/fresh/cold final receipt validators are reused. The two new
   producer finals use schema `ferrule.storage-demand.paired-read-quality-producer.v1`
   with actual label/report/executable hashes, argv, production source digest and
   supervisor hash. They must come from the maintained authenticated producer,
   not from CPU preregistration. Actual helper/worker PID/birth/parent and launch
   relation must match a successfully drained supervisor receipt.
4. `runtime_pin_manifest` must include actual loader/native runtime pins, no
   unknown NCCL binding. Ready and after-request actual `/proc/PID/maps` are checked
   by the maintained worker. Help must belong to the exact staged CLI and expose
   **all** literal read-pipeline flags. Independently pinned numerical references,
   tokenizer and checkpoint metadata must match the frozen oracles. The candidate
   input supplies pinned approved local `tokenizer_files` copies keyed by
   `tokenizer.json` and `tokenizer_config.json`; no CPU NAS probe is performed. Source-path
   relocation/clone offsets are handled through original absolute-key source
   manifests; no mismatched native/compiler/runtime content is excused.

The new build is opt3/fat-LTO/codegen1/strip symbols and NVTX-capable in **both**
arms. Ordinary timing has explicit `FERRULE_PROFILE_NVTX=0`; optional profiling
has `=1`. Existing DEBUG/TRACE and sampling overhead still exists and is unmeasured.
No zero-overhead or historical-binary timing comparison is claimed.

## Optional profiling — separate, at most one paired block

The preregistered optional order is Blocking then Lookahead after **all ten ordinary
lifecycles finish**. It would use the same CLI/config/schedule (24 additional
requests, 10 measured QA diagnostic slots), never mix profile timing into the
ordinary statistics and require 2*650+100 s remaining in the **same** two-hour
absolute envelope. Insufficient time means unissued, not a shortened trace.

**There is deliberately no profiling launch entry in this stage.** The maintained
TP1 capture enforces a direct-child server relation; naively inserting `nsys`
would invalidate that custody contract. The maintained official authenticated
collector adapter must receive a separately reviewed TP1 binding before profiling
can run. `profile_audit.py` supplies a CPU-only graph-expansion layer over the
maintained bidirectional API/activity auditor; it retains missing loss/coverage
as unverified and never converts correlation closure into full trace acceptance.
Require pinned official nsys/help/version, complete `.nsys-rep` and SQLite exports,
actual PID/UUID and request-window bindings, successful collector descendant
custody, independent collection coverage and loss/diagnostic receipts. Graph
launch expansion must use actual exported node relations, not assume one kernel
per launch. NVTX exports fixed names and unsigned numeric fields only, never
prompts/text/paths/env; host ranges do not prove GPU overlap.

## CPU reproduction and later owner handoff

From repository root (base Python 3.12; `-S -B`, no site hooks/bytecode):

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/paired_read_v1 -p 'test_*.py' -v
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/protocol.py plan
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/protocol.py check
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/run_paired.py plan
```

The exclusive CPU seal command (already used for `cpu-sealed.json`; do not
replace it) is `protocol.py seal-cpu --output
papers/arxiv/experiments/storage_demand/paired_read_v1/cpu-sealed.json` with the
same Python prefix. If code/workflow/source changes, an explicitly reviewed new
seal filename is required; hashes fail closed.

After actual GPU gates PASS, the owner supplies the input descriptor and invokes
`protocol.py bind-candidate --candidate OWNER_INPUT --output NEW_CANDIDATE_SEAL`,
then `run_paired.py check --candidate NEW_CANDIDATE_SEAL`. Only the owner can invoke
`run_paired.py run --candidate NEW_CANDIDATE_SEAL --output NEW_ARTIFACTS_DIRECTORY
--gpu-uuid SELECTED_PHYSICAL_UUID --eight-uuids EIGHT_ORDERED_PHYSICAL_UUIDS
--sole-gpu-owner-granted --owner-note REVIEWED_NOTE`. These uppercase arguments
are handoff requirements, not fabricated values or commands executed here.

Actual CLI build reproduction is recorded by its maintained CPU builder as
`cargo build --locked --offline --release -p ferrule-cli --no-default-features
--features cuda,nvtx-profile --message-format=json` in the read-only archived source
with a new target directory and the exact isolated pinned allowlist. Existing
quality selectors in `bf16_full40.rs` and `bf16_fresh_quality/mod.rs` provide the
Blocking/Lookahead full40/fresh/ordinary/cold gates; the constructor selector is
in `crates/ferrule-model/tests/expert_read_constructor.rs`. Do not run ignored CUDA
selectors, a build/help probe or query merely to prepare this CPU protocol.

Dependency and reproduce manifests live in the actual build's pinned compiler,
CUTLASS, compiled-dependency, native archive and runtime receipts. Archive only
approved source/dependency evidence, command receipts, stats and protocol bindings;
**no weights tarball, arbitrary environment dump or ENV secrets**. Future artifacts
must stay in approved owner-local custody; token IDs/raw SSE can reconstruct text
and mountstats can expose export paths. Shared NFS client READ RPC/serverreadbytes
deltas never become NAS disk/physical network measurements.

## Main-agent small stage

Stage only this directory (including dedicated tests and CPU seal/validation).
No changes to production, manuscript, histories, baseline prepared/cpu-run data or
shared runner helpers are required. `/papers/` is ignored by default, so main
uses a path-scoped `git add -f` for this directory, never `git add -A`. Main coordinates the commit/push. Suggested
subject: `Preregister paired blocking and lookahead study`.
