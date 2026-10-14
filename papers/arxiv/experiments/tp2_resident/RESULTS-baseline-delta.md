# TP2 observer baseline-delta rerun

## Final status

The observer fix is implemented and tested, but the new GPU reruns did not reach
readiness, so no Hello2 or canary request was issued and QA8 was correctly
blocked. Prior failed canary receipts remain unchanged.

- `tp2-resident-20261008-003`: stopped after **12.1608 s** because an
  `nvidia-smi` query exceeded its three-second timeout. No delta violation;
  authenticated drain and all-GPU release passed.
- `tp2-resident-20261008-004`: a TP worker logged `CUDA driver initialization
  failed, you might not have a CUDA gpu.` The observer recorded selected 3 MiB
  peaks and no nonselected delta violation. The inherited v2 PSS helper also
  exhausted its ESRCH retries while reading child process memory. Startup
  stopped after **32.3163 s**. Causal ordering of the CUDA failure and observer
  shutdown is not established by these logs. No request was issued.
  Authenticated drain and all-GPU release passed.

These are startup failures, not canary PASS/FAIL results. Run 004 is an explicitly
separate observer revision following 003; neither case automatically relaunched
a server. No transport fallback, CPU offload, EP, Ferrule launch, inference-source
or model change, or foreign PID signal was used.

| Run | Sampled GPU0 / GPU1 high water | Tree PSS high water | Supervisor wall | Launch ID |
|---|---:|---:|---:|---|
| 003 | 0 / 0 MiB | 0.9330 GiB | 27.9836 s | `c569bfa823284cb9aadf8da78c0c38d2` |
| 004 | 3 / 3 MiB | 4.5630 GiB | 49.2111 s | `40304c5aaa6f4af0b0b41308866d0300` |

These high-water values cover only successful samples before startup failed;
a missing sample is not a zero-memory observation. No TTFT/E2E or throughput
measurement exists for these runs.

All eight pre-launch baselines and final observations were **0 MiB**, utilization
0, no compute apps. Successful runtime samples showed no positive nonselected
delta in either run. The 1 MiB behavior is regression-tested, not newly observed
in these failed starts. Both selected locks were reacquired for the final idle
check and then released.

The NCCL/P2P environment remains pinned (`NCCL_P2P_DISABLE=1`, INFO logging),
and the live topology still reports P2P `CNS`. Neither new run reached model
loading, so **no new effective Marlin/GDN or collective-channel evidence** exists.
Prior runs' SHM/Marlin/GDN observations are historical, not borrowed as new proof.

## Observer change

`sampling_v4.py` snapshots all eight physical UUIDs from the verified pre-launch
idle baseline and seals that map into the worker launch receipt. Each sample
records baseline bytes, observed bytes, signed delta, positive delta, utilization,
PID evidence, and tolerance. The explicit nonselected tolerance is **1 MiB**.
Only an unexplained positive nonselected memory delta **greater than 1 MiB**
rejects. A stable 1 MiB residue is retained as evidence rather than treated as
absolute-idle failure. Missing samples are never converted to zero.

The existing v2 ESRCH handling remains unchanged. v4 adds only one bounded
retry for a three-second `nvidia-smi` timeout and records each timeout. Other
errors fail closed. Pre/post idle gates still require all eight GPUs to have no
compute apps, zero utilization, and selected GPUs at zero memory; no foreign
process is signaled.

## Immutable evidence

- [Machine-readable summary](RESULTS-baseline-delta.json)
- [v4 runner](experiment_v4.py)
- [v4 observer](sampling_v4.py)
- [v4 observer tests](test_sampling_v4.py)
- [Run 003 execution](../runs/tp2-resident-20261008-003/execution.json)
- [Run 004 execution](../runs/tp2-resident-20261008-004/execution.json)
- [Run 004 worker log](../runs/tp2-resident-20261008-004/hello2/server.log)
- [Run 004 baseline](../runs/tp2-resident-20261008-004/hello2/baseline.json)
- [Run 004 sampled deltas](../runs/tp2-resident-20261008-004/hello2/samples.jsonl)
- [Final idle recheck](final-owner-recheck-baseline-delta.json)
- [Final release](final-owner-release-baseline-delta.json)

The same TP2/no-EP/no-CPU-offload vLLM flags, CUDA12.9 JIT/runtime pins,
selected UUIDs, NCCL/P2P environment, and frozen manifest
`102c798baeb5200bc8ed8386eabed6146870b3b6da2a90fc6010677041352c7e` were
preserved. Tests: **42 original + 28 TP2/observer passed**.

## Gated workload and limits

The original three canaries and expected first-eight-token text/usage are
unchanged; no HF regeneration. If a future run passes all three, the declared
order slots are `F1,V1` then `V2,F2`: F slots are unavailable because Ferrule TP2
is absent; each V slot is a fresh-process Japan QA8 block with one excluded first
request, two warmups, and five measured requests. These attempts never entered
those blocks. `fairmatch=false`, `exact_speedup=null`; no extrapolation.

Limits remain startup 900 s, request 120 s, teardown 120 s, worker supervision
1600 s with the existing cleanup/idle reserve below the 1700 s ceiling. The
128 GiB PSS and 22 GiB/card observational ceilings are not hard physical caps.
The runtime/source/reference and prior-run pins were rechecked after execution.
M7/G10 and runs 001/002/003 receipts were not changed.
