# Ferrule throughput methodology

<!-- markdownlint-disable MD013 MD060 -->

This document defines how Ferrule models, measures, and reports inference
throughput. It separates acceptance targets and analytical ceilings from
end-to-end measurements.

## 1. Reporting contract

Every performance number must be labeled as exactly one of:

| Class | Meaning |
|---|---|
| **Target** | An acceptance threshold, not evidence that it has been reached. |
| **Model** | A roofline, budget, or lower-bound calculation based on stated assumptions. |
| **Measurement** | A result from a named command, workload, build, device, and artifact. |

Ferrule's headline metric is:

```text
externally committed output tokens / end-to-end wall-clock second
```

Only tokens visible to the caller after exact commit count. Draft tokens,
target rows, accepted-prefix candidates, internal cycles, and warmup tokens do
not count unless they are externally committed.

A valid report includes:

- hardware, driver, CUDA, and software profile;
- exact checkpoint and Git revision;
- prompt and output lengths;
- request count, concurrency, and request-rate policy;
- warmup policy;
- successful, failed, cancelled, and timed-out requests;
- externally committed tokens and wall time;
- proposal, acceptance, correction, and target-row accounting;
- physical read, upload, install, and publication counters;
- resource capacities, high-water marks, and shutdown residuals;
- raw artifact paths.

## 2. Acceptance target

The current CUDA-profile target is a warm externally committed output
throughput whose 95% confidence lower bound reaches:

```text
16 output tok/s
```

This is a target, not a current result. Ferrule makes no production-readiness or
state-of-the-art throughput claim until correctness, lifecycle, resource, and
performance evidence pass on the same revision and workload.

## 3. Release equation

For one exact proposal-verification cycle, let:

| Symbol | Meaning |
|---|---|
| $Q$ | target verification rows in the cycle |
| $C(Q)$ | mean externally committed output tokens from the cycle |
| $T_{\mathrm{draft}}(Q)$ | proposal time |
| $T_{\mathrm{verify}}(Q)$ | exact target verification time |
| $T_{\mathrm{commit}}(Q)$ | commit, rollback, correction, or bonus staging time |
| $T_{\mathrm{overlap}}(Q)$ | safe overlap proven by a causal timeline |

Then:

$$
T_{\mathrm{cycle}}(Q) = T_{\mathrm{draft}}(Q)
  + T_{\mathrm{verify}}(Q)
  + T_{\mathrm{commit}}(Q)
  - T_{\mathrm{overlap}}(Q),
$$

and:

$$
R_{\mathrm{output}}(Q) = \frac{C(Q)}{T_{\mathrm{cycle}}(Q)}.
$$

For example, four externally committed tokens from a complete 250 ms cycle
imply:

$$
R_{\mathrm{output}} = \frac{4}{0.25\ \mathrm{s}} = 16\ \mathrm{tok/s}.
$$

This example explains the target; it is not a Ferrule measurement. Multiplying
a single-row target rate by accepted draft length is generally invalid because
compute, memory traffic, expert union, and I/O all depend on $Q$.

## 4. Resource roofline

For one exact target pass or cycle, define:

| Symbol | Meaning |
|---|---|
| $F_{\mathrm{gpu}}, F_{\mathrm{cpu}}$ | operations executed by GPU and CPU |
| $B_{\mathrm{mem}}$ | bytes crossing the relevant memory system |
| $B_{\mathrm{nvme}}$ | bytes read from storage |
| $B_{\mathrm{intra}}, B_{\mathrm{inter}}$ | intra-node and inter-node communication bytes |
| $P_r$ | peak compute rate of resource $r$ |
| $W_r$ | peak bandwidth of resource $r$ |
| $\eta_r \in (0,1]$ | measured efficiency of resource $r$ |
| $C_r L_r$ | latency-sensitive communication rounds and per-round latency |
| $T_{\mathrm{serial}}$ | unavoidable sequential or control time |

Per-resource lower bounds are:

$$
T_{\mathrm{gpu}} = \frac{F_{\mathrm{gpu}}}{\eta_{\mathrm{gpu}}P_{\mathrm{gpu}}},
\qquad
T_{\mathrm{cpu}} = \frac{F_{\mathrm{cpu}}}{\eta_{\mathrm{cpu}}P_{\mathrm{cpu}}},
$$

$$
T_{\mathrm{mem}} = \frac{B_{\mathrm{mem}}}{\eta_{\mathrm{mem}}W_{\mathrm{mem}}},
\qquad
T_{\mathrm{nvme}} = \frac{B_{\mathrm{nvme}}}{\eta_{\mathrm{nvme}}W_{\mathrm{nvme}}},
$$

$$
T_{\mathrm{intra}} =
  \frac{B_{\mathrm{intra}}}{\eta_{\mathrm{intra}}W_{\mathrm{intra}}}
  + C_{\mathrm{intra}}L_{\mathrm{intra}},
$$

$$
T_{\mathrm{inter}} =
  \frac{B_{\mathrm{inter}}}{\eta_{\mathrm{inter}}W_{\mathrm{inter}}}
  + C_{\mathrm{inter}}L_{\mathrm{inter}}.
$$

With perfect overlap, the optimistic latency floor is:

$$
T_{\mathrm{ideal}} = \max(
  T_{\mathrm{gpu}},
  T_{\mathrm{cpu}},
  T_{\mathrm{mem}},
  T_{\mathrm{nvme}},
  T_{\mathrm{intra}},
  T_{\mathrm{inter}},
  T_{\mathrm{serial}}
).
$$

With no overlap:

$$
T_{\mathrm{no\ overlap}} =
  T_{\mathrm{gpu}} + T_{\mathrm{cpu}} + T_{\mathrm{mem}}
  + T_{\mathrm{nvme}} + T_{\mathrm{intra}} + T_{\mathrm{inter}}
  + T_{\mathrm{serial}}.
$$

Real execution lies between these models and may be slower because of queueing,
dependencies, launch overhead, cache misses, and contention. A roofline must
never be labeled as measured throughput.

## 5. Shared-memory accounting

On coherent-memory systems, CPU, GPU, CUDA copies, pinned staging, and storage
DMA compete for the same physical capacity and memory bandwidth. On discrete
systems, host, interconnect, and device traffic still need separate accounting.
No component may independently claim the full vendor bandwidth when traffic
shares a bottleneck.

A useful byte identity is:

$$
B_{\mathrm{mem}} = B_{\mathrm{dense}} + B_{\mathrm{active\ expert}}
  + B_{\mathrm{KV}} + B_{\mathrm{head}} + B_{\mathrm{storage\ destination}}
  + B_{\mathrm{staging/frame}} + B_{\mathrm{temporary}}.
$$

Advertised accelerator peak FLOP/s is not a sustained small-batch MoE rate.
Every roofline needs measured efficiency for the exact shape, dtype, and
operator boundary.

## 6. Routed-expert I/O budget

For the current DeepSeek-V4 profile:

- routed layers: 43;
- selected experts per row and layer: 6;
- one expert payload: approximately 12.75 MiB.

The all-cold payload for one row is:

$$
B_{\mathrm{routed,cold}}
  = 43 \cdot 6 \cdot 12.75\ \mathrm{MiB}
  \approx 3.21\ \mathrm{GiB/pass}.
$$

If a measured storage bandwidth is $W_{\mathrm{nvme}}$, the storage-only
all-cold ceiling is:

$$
R_{\mathrm{nvme,cold}}
  \le \frac{W_{\mathrm{nvme}}}{3.21\ \mathrm{GiB/pass}}.
$$

This is a model, not end-to-end throughput.

For $Q$ verification rows, let $E_{\ell,q}$ be the exact selected-expert set for
row $q$ at layer $\ell$, and let $S_e$ be one expert payload. The all-cold route
union is:

$$
B_{\mathrm{routed,cold}}(Q)
  = S_e \sum_{\ell=1}^{43}
    \left|\bigcup_{q=1}^{Q}E_{\ell,q}\right|.
$$

Therefore:

$$
3.21\ \mathrm{GiB}
  \le B_{\mathrm{routed,cold}}(Q)
  \le 3.21Q\ \mathrm{GiB},
$$

until each layer's union reaches its expert count. The lower bound requires
perfect route reuse; the upper bound represents disjoint selected sets. Real
route traces, not an independence assumption, must supply the release
calculation.

For desired output throughput $R^*$, mean committed tokens $C$, and measured
storage bandwidth $W_{\mathrm{nvme}}$, the per-cycle read budget is:

$$
B^*_{\mathrm{NVMe/cycle}}
  \le \frac{C W_{\mathrm{nvme}}}{R^*}.
$$

Meeting the target requires a cache-heavy regime, useful route reuse, or proven
overlap. Increasing queue depth does not change the byte budget.

## 7. Overlap and uncovered critical path

Concurrency is not proof of overlap. Ferrule counts overlap only when a causal
timeline establishes that:

1. transaction A is blocked on a real read, upload, CUDA event, publication, or
   exact dependency;
2. transaction B performs useful ready work during the same interval;
3. both retain valid hard-resource custody and transaction ownership;
4. completion and wake timestamps reconcile with the resource ledger.

For a wait interval $W$ and useful-work interval union $U$:

$$
T_{\mathrm{uncovered}} = |W \setminus U|.
$$

Shared waits are unioned rather than multiplied by waiter count. Queueing caused
by exhausted hard credits is reported separately from physical service time.

Required stage accounting includes:

- reserve and fair-dispatch wait;
- read submission, bytes, CQE, and host-ready time;
- upload submission, bytes, CUDA event, and frame-ready time;
- install, publication, and targeted wake;
- continuation ready, resume, completion, and detach;
- uncovered read, upload, wake, and resume duration.

## 8. Hard-resource evidence

A valid run reports capacity, high-water mark, acquisition wait, release, and
shutdown residual for every configured hard resource. These include storage
entries, slabs, bytes, uploads, frames, leases, load operations, execution
workspace, KV state, continuations, runnable cohorts, and transaction execution
ownership.

Names in serialized output are schema-controlled; use the emitted schema rather
than reconstructing counter names from this document. Every high-water mark
must remain at or below capacity, and all shutdown residual grants must be zero.

## 9. Reproducible commands

### Build

Build for the detected device:

```bash
just cuda-info
just cutlass-setup
just build-cuda
```

Compile an explicit target without running it:

```bash
just check-cuda-arch sm_103
```

### Resident-driver probe

```bash
just dsv4-runtime-driver-bench \
  "Hello" \
  "Explain Ferrule in one sentence." \
  8 \
  2 \
  4096 \
  43
```

Freeze all non-default arguments and the output artifact path in the report.

### OpenAI-compatible serving sweep

Terminal 1:

```bash
just dsv4-serve
```

Terminal 2:

```bash
just dsv4-vllm-bench sweep
```

Preserve results for each tested concurrency and report request success and
failure alongside throughput.

### Validation lanes

```bash
FERRULE_NO_CUDA=1 just lint
FERRULE_NO_CUDA=1 just test
just test-cuda-required
just check-cuda-arch sm_103
just deny
```

Use repository recipes so the detected or explicit CUDA target reaches both
Cargo and NVCC. Ferrule uses standard Cargo plus NVCC and a pinned CUTLASS
checkout.

## 10. Artifact checklist

Keep these artifacts together for each accepted run:

- command line and environment overrides;
- Git revision and dirty-state report;
- stdout JSON and stderr log;
- wall time and process exit code;
- device, driver, CUDA, filesystem, and storage identity;
- checkpoint identity;
- materialization and driver statistics;
- operator counters and acceptance distribution;
- stage bytes, counts, time, and uncovered critical path;
- hard-resource capacity, high-water, and residual table;
- request-level success and failure output;
- profiler or timeline evidence used to claim overlap.

A timeout, protocol failure, nonzero shutdown ownership, or missing committed
output accounting is diagnostic evidence, not an accepted throughput result.

## 11. Common reporting errors

Do not:

- present 16 tok/s as achieved before a frozen acceptance run passes;
- convert one successful request into a throughput claim;
- report internal cycle rate as externally committed output throughput;
- infer overlap from concurrency, utilization, or lower wall time alone;
- multiply a single-row rate by accepted length without width-dependent work;
- divide logical requested bytes by time when exact-key dedup changes physical
  bytes;
- omit failed, cancelled, or timed-out requests;
- use vendor peak FLOP/s as a measured small-batch rate;
- extrapolate one CUDA profile to another accelerator;
- use an old benchmark as evidence for the current revision.
