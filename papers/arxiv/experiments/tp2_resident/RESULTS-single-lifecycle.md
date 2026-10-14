# TP2 resident vLLM single-lifecycle result

## Result

`tp2-resident-single-lifecycle-20261009-001` completed with exactly one server
start. The same server handled Hello2, all three frozen canaries, and QA8. No
Ferrule TP2 process was started: an admitted Ferrule TP2 serving profile was
absent in that experiment, so no fair paired block, speedup, or extrapolation is
claimed. The later test-only progress below does not change this historical run.

- vLLM profile: TP2/PP1/DP1, no EP, explicit CPU offload 0
- selected UUIDs: `GPU-b980f22b-58a3-fc88-21c1-ae86053d8535`,
  `GPU-63f1bd01-ddf9-a742-ee48-9389892493d9`
- startup: **65.7325 s**
- selected GPU peak: **19.2842 GiB/card**
- server-tree PSS peak: **6.8787 GiB**; RSS peak **8.5697 GiB**
- complete host samples: 31; one incomplete leaf sample was recorded with
  missing PSS and did not terminate the server; 32 GPU samples, no missing NVML
  attempts
- authenticated supervisor wall: **88.1403 s**, ECHILD/drain proven

## Requests

Hello2 returned `, I` with 1 input and 2 output tokens:

- TTFT **230.452 ms**
- E2E **311.259 ms**

The three canaries matched the original manifest-derived expected visible text
and output usage exactly, with no HF regeneration:

| Canary | Input tokens | Output tokens | Expected/observed visible prefix |
|---|---:|---:|---|
| Japan | 5 | 8 | ` Tokyo.\nThe capital of Japan is` |
| arithmetic | 5 | 8 | ` ?\n\n7 + 8 = ` |
| Chinese | 3 | 8 | `北京，北京是中国的首都。那么` |

This is a visible text + usage gate only. Token IDs, logits, routing, state, and
full numerical equivalence were not observed.

QA8 used the same server and the Japan prompt: one excluded first QA-phase
request, two warmups, and five measured requests. All had 5 input and 8 output
tokens. The five measured requests had:

- mean TTFT: **130.657 ms**
- p50/p95 TTFT: **130.195 / 132.551 ms**
- mean E2E: **703.337 ms**
- p50/p95 E2E: **701.390 / 714.888 ms**
- all 5 successful, finish reason `length` with natural EOS policy retained, no request failures

## Observer and runtime evidence

The current shared observer:

- binds the root PID/start time/parent once;
- omits disappeared descendant leaves/branches and records incomplete samples;
- preserves `pss_bytes: null` for incomplete samples rather than zeroing them;
- fails only on root custody loss, no valid complete sample by the 60-second
  coverage deadline, or resource/delta policy violation;
- compares all eight GPUs against the pre-launch UUID baseline and accepts only
  nonselected positive deltas up to the explicit 1 MiB tolerance;
- performs at most one 15-second diagnostic NVML query pair after a normal query
  timeout, recording the missing attempt in the sample stream.

The one incomplete sample omitted PID 15844 because `/proc/15844/smaps_rollup`
disappeared. The authenticated root remained valid; the server continued and
later complete samples were collected.

Runtime logs show:

- `NCCL_P2P_DISABLE=1`, NCCL 2.29.7+CUDA12.9
- local NCCL channels `via SHM/direct/direct`
- `MarlinFP8ScaledMMLinearKernel` and MARLIN FP8 MoE
- Triton/FLA GDN prefill and Triton GDN decode fallback because the fused decode
  op was not built
- model load memory 17.04 GiB; available KV cache 1.7 GiB
- no CPU-offload contradiction and no expert-parallel flag

The post-run process-map sample observed CUDA12/NCCL libraries from the pinned
vLLM environment plus CUDA13.2 runtime/NVRTC libraries available in the process
maps. This is retained as a dynamic-library closure caveat; no library or
runtime environment was changed, and it is not asserted as the cause of the
historical run-004 failure.

## Historical failure comparison

Runs 001–004 have identical declared launch environment, argv, UUIDs, staged
entrypoint hash, and runtime inventory. Run 003 was observer-interrupted during
Torch/FLA lazy CUDA initialization after an NVML query timeout. Run 004 logged
a TP worker failure at `gpu_worker.py:379` →
`torch.accelerator.set_device_index` → `torch._C._cuda_init()` with
`RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu.`
It failed before NCCL initialization; no numeric CUDA driver error was emitted.
The successful current run provides readiness evidence but does not retroactively
prove the unresolved driver root cause.

## Release and validation

A final independent all-eight-GPU idle check passed: 0 MiB, 0% utilization, no
compute apps. GPU0/GPU1 locks were reacquired for that check and released. No
foreign PID was signaled.

- 42 original tests passed.
- 37 current observer/contract tests passed.
- source, runtime, reference, driver, and prior-run pins rechecked.
- M7/G10 histories and failed canary receipts remain unchanged.
- `fairmatch=false`, `exact_speedup=null`.

Machine-readable receipt: [`RESULTS.json`](../runs/tp2-resident-single-lifecycle-20261009-001/RESULTS.json).

## Direct artifacts

- [Exact historical environment and worker-init comparison](../runs/tp2-resident-single-lifecycle-20261009-001/comparison.json)
- [Sealed protocol/source/runtime/driver pins](../runs/tp2-resident-single-lifecycle-20261009-001/prepared.json)
- [Server log](../runs/tp2-resident-single-lifecycle-20261009-001/session/server.log)
- [Three-canary gate](../runs/tp2-resident-single-lifecycle-20261009-001/session/canary8/gate.json)
- [QA8 full HTTP receipt](../runs/tp2-resident-single-lifecycle-20261009-001/session/qa8-V1/http.json)
- [Loaded-library mappings and post-observation hashes](../runs/tp2-resident-single-lifecycle-20261009-001/loaded-library-evidence.json)
- [Final authenticated execution receipt](../runs/tp2-resident-single-lifecycle-20261009-001/execution.json)
- [Final lock reacquisition/idle](../runs/tp2-resident-single-lifecycle-20261009-001/final-owner-recheck.json)
- [Final release](../runs/tp2-resident-single-lifecycle-20261009-001/final-owner-release.json)

Launch ID: `d826244b213947d4b5ce2843f1537961`; server PID 14168; TP worker
PIDs 15060/15061 (local process identities, not a claimed host-NVML PID mapping).
All nonselected GPU sampled deltas were zero; baseline and final memory were
zero on all eight devices. The explicit 1 MiB tolerance was not exercised by
this run. Planning remains 19.2 GiB/card and the observational ceiling 22 GiB/card;
19.2842 GiB observed is above planning but below the declared ceiling.

Five measured QA8 requests took 3.536264 s client wall for 40 actual output tokens.
This denominator includes client overhead; no larger-load, longer-output, cold
replica, or Ferrule speedup inference is made. The server/JIT/page caches were
not cleared, so the shorter startup than old attempts is not an optimization claim.


## Later Ferrule progress — CPU receipt audit, 2026-10-09

The shared/dense separate gate/up fix has now passed the claim-aware physical
**PP1/TP2/EP1 GPU0/1** experimental gates. The tiny lifecycle and full40 test use
the same pinned **test binary**, SHA256
`63d5989ca176ed49bd1c787418321c83c3b842f2df6c3f701cd4fd77ea384f1c`;
the path named `staged/ferrule` is **not a public serving CLI**.

| Full40 comparison | Finite logits compared | Max absolute error / failures |
|---|---:|---:|
| Hello prefill | 248,320 | 0 / 0 |
| Hello first decode | 248,320 | 0 / 0 |
| Capital prefill | 1,241,600 | 0 / 0 |
| Capital first decode | 248,320 | 0 / 0 |
| Total | **1,986,560** | **0 / 0** |

All comparisons retain `atol=0.00025`, `rtol=0.012`, with zero nonfinite values
and zero finite RMSE. Both ranks report completed preparation claims and
quiescent shutdown without device/operator quarantine. Saved receipts show
exit 0, authenticated ECHILD/drain, before/after idle and lock/owner release.
Full40 supervisor wall was 334.1169 s (not serving startup or request latency);
sampled GPU peak was 19,372,441,600 bytes/card and tree PSS 53,374,798,848 bytes.
These are **test lifecycle resources**, not new serving-performance measurements.
GPU PID attribution remains unknown; 4 incomplete host samples remain missing.

The CPU audit rehashed both binaries/logs, prepared provenance/runtime pins,
model handoff and three fixed model source files; no GPU query or numerical
rerun was performed. Authentication matches launch/helper and worker
PID/birth/parent; the custodian's pre-`setsid` root snapshot and authenticated
worker session fields differ as expected, not a different root process.
Supervisor GPU quiescence itself is unverified; idle evidence comes from the
separate saved snapshots, not from treating ECHILD as a GPU-idle proof.

- [Full40 result](../../../../target/validation/tp2-mixed-claim-gates/shared-full40-01/result.json)
- [Full40 log](../../../../target/validation/tp2-mixed-claim-gates/shared-full40-01/test.log)
- [Tiny lifecycle result](../../../../target/validation/tp2-mixed-claim-gates/shared-tiny-01/result.json)
- [CPU audit and hashed receipt inventory](claim-gates-audit-20261009.json)

**At this earlier checkpoint, blocked:** all-boundary/state and multistep acceptance, public serving
CLI/profile, same-serving-binary quality/lifecycle binding, and fresh F/V
paired blocks. The independent HF-eager logical TP2 oracle (`fb96ba4d…caa71d`)
is not vLLM Marlin arithmetic and is distinct from the original visible-canary
manifest (`102c798b…352c7e`). E2 is incomplete; `fairmatch=false`,
`exact_speedup=null`. The vLLM 0.28.0+cu129 / Torch 2.13.0+cu129 TP2-noEP eager
context128/batch8/seq2 QA8 **11.31 tok/s** remains a five-request descriptive
observation, not an optimal setting, statistical conclusion or extrapolation.


## Full-boundary comparator update — CPU receipt audit

`native-whole-capture-full40-02` is now paired with
`papers/arxiv/experiments/tp2_mixed/oracle/runs/native-whole-capture-comparison-01.json`
in [full-boundary-audit.json](full-boundary-audit.json). Hello and Capital each
contain **2,178/2,178 exact raw/comparator fields**, with zero failures, zero
maximum absolute error and matching greedy IDs for prefill and first decode.
Raw/canonical payload hashes, the staged test binary, supervisor ECHILD/drain,
before/after idle and release receipts were rechecked without rerunning CUDA.
This closes the bounded full-boundary two-call comparator scope; independent
multistep TP2 acceptance was still pending at that checkpoint; see the subsequent closure below.

At that checkpoint the catalog had `PUBLIC_QUALITY_ACCEPTED=false`; that receipt alone did
not open the public Ferrule serving profile. The future pair remains descriptive
only (`fairmatch=false`, `exact_speedup=null`) and must use the actual catalog
profile/flags, not an invented strict or offload flag.


## Subsequent release-serving closure and fresh F/V pilot

Full40 Hello/Capital each 2,178 exact fields plus all six lifecycle recipes
(33 calls / 35,937 exact metrics) are now closed by fresh `greedy4-02`;
old failed supervisor receipts are preserved. Catalog gate opened and the actual
release `e017a3a8…26c1b0b` passed two authenticated serving HTTP smoke lifecycles.
The **[fresh F1,V1 / V2,F2 pilot](RESULTS-serving-pilot.md)** then completed:
Ferrule QA8 5.5213 / 5.5414 tok/s, vLLM 10.3966 / 10.5621 tok/s, five requests
per fresh server, all canaries pass. This does not rewrite the historical
11.31 tok/s baseline or infer parity/speedup. E2 incomplete, `fairmatch=false`,
`exact_speedup=null`; all six new servers drained and final all-eight idle/release
verified. CPU collective/NCCL, Mixed/Marlin and serial-decode differences remain.
