# Ferrule throughput methodology

This document defines how Ferrule models, measures, and reports inference throughput. It separates
release targets from analytical ceilings and both from end-to-end measurements.

## 1. Reporting contract

Every performance number must be labeled as exactly one of:

| Class | Meaning |
|---|---|
| **Target** | An acceptance threshold. It is not evidence that the threshold has been reached. |
| **Model** | A roofline, budget, or lower-bound calculation based on stated assumptions. |
| **Measurement** | A result produced by a named command, workload, build, device, and artifact. |

Ferrule's headline metric is:

```text
externally committed output tokens / end-to-end wall-clock second
```

Only tokens visible to the caller after exact commit count toward output throughput. Draft tokens,
target rows, accepted-prefix candidates, internal cycle counters, and warmup tokens are not output
tokens unless they are externally committed.

A valid report includes at least:

- hardware and software profile;
- exact checkpoint identity;
- prompt/output lengths and request count;
- concurrency and request-rate policy;
- warmup policy;
- successful and failed requests;
- externally committed tokens and wall time;
- acceptance and target-row accounting;
- physical read/upload/publish counters;
- resource capacity and high-water marks;
- cancellation, stale, failure, and shutdown residuals;
- raw artifact paths.

## 2. Current target and evidence boundary

The current GB10 acceptance target is a warm externally committed output throughput whose 95%
confidence lower bound reaches:

```text
16 output tok/s
```

This is a **target**, not a current result.

Current evidence boundaries as of 2026-07-29:

- a real GB10 `n1` run previously completed in `12.11 s`;
- that completion proves one historical end-to-end path ran, not that the 16 tok/s target was met;
- the latest real GB10 `n8` attempt timed out after `91.33 s` with empty stderr;
- the current wake-path patch still needs build, unit-test, and real-device revalidation;
- `c1`, `c2`, and `c4` overlap/counter acceptance has not completed;
- Ferrule therefore makes no release, production-readiness, or SOTA throughput claim.

The authoritative implementation status is maintained in [ROADMAP.md](ROADMAP.md).

## 3. Release equation

For one exact DSpark cycle, let:

| Symbol | Meaning |
|---|---|
| $Q$ | target verification rows in the cycle |
| $C(Q)$ | mean externally committed output tokens produced by the cycle |
| $T_{\mathrm{draft}}(Q)$ | proposal time |
| $T_{\mathrm{verify}}(Q)$ | exact target verification time |
| $T_{\mathrm{commit}}(Q)$ | commit, rollback, correction, or bonus staging time |
| $T_{\mathrm{overlap}}(Q)$ | safe overlap proven by a timeline, not inferred from concurrency |

Then:

$$T_{\mathrm{cycle}}(Q) = T_{\mathrm{draft}}(Q) + T_{\mathrm{verify}}(Q) + T_{\mathrm{commit}}(Q) - T_{\mathrm{overlap}}(Q),$$

and:

$$R_{\mathrm{output}}(Q) = \frac{C(Q)}{T_{\mathrm{cycle}}(Q)}.$$

For example, four externally committed tokens from a complete 250 ms cycle imply:

$$R_{\mathrm{output}} = \frac{4}{0.25\ \mathrm{s}} = 16\ \mathrm{tok/s}.$$

This example explains the target's meaning. It is not a Ferrule measurement. It is generally wrong
to multiply a single-row target-pass rate by accepted draft length: compute, memory traffic, expert
route union, and I/O all depend on $Q$.

## 4. Resource roofline

For one exact target pass or cycle, define:

| Symbol | Meaning |
|---|---|
| $F_{\mathrm{gpu}}, F_{\mathrm{cpu}}$ | operations executed by GPU and CPU |
| $B_{\mathrm{um}}$ | bytes crossing the shared memory system |
| $B_{\mathrm{nvme}}$ | bytes read from storage |
| $B_{\mathrm{intra}}, B_{\mathrm{inter}}$ | intra-node and inter-node communication bytes |
| $P_r$ | peak compute rate of resource $r$ |
| $W_r$ | peak bandwidth of resource $r$ |
| $\eta_r \in (0,1]$ | measured efficiency of resource $r$ |
| $C_r L_r$ | latency-sensitive communication rounds and per-round latency |
| $T_{\mathrm{serial}}$ | unavoidable sequential/control time |

Per-resource lower bounds are:

$$T_{\mathrm{gpu}} = \frac{F_{\mathrm{gpu}}}{\eta_{\mathrm{gpu}}P_{\mathrm{gpu}}}, \qquad
T_{\mathrm{cpu}} = \frac{F_{\mathrm{cpu}}}{\eta_{\mathrm{cpu}}P_{\mathrm{cpu}}},$$

$$T_{\mathrm{um}} = \frac{B_{\mathrm{um}}}{\eta_{\mathrm{um}}W_{\mathrm{um}}}, \qquad
T_{\mathrm{nvme}} = \frac{B_{\mathrm{nvme}}}{\eta_{\mathrm{nvme}}W_{\mathrm{nvme}}},$$

$$T_{\mathrm{intra}} = \frac{B_{\mathrm{intra}}}{\eta_{\mathrm{intra}}W_{\mathrm{intra}}} + C_{\mathrm{intra}}L_{\mathrm{intra}},$$

$$T_{\mathrm{inter}} = \frac{B_{\mathrm{inter}}}{\eta_{\mathrm{inter}}W_{\mathrm{inter}}} + C_{\mathrm{inter}}L_{\mathrm{inter}}.$$

With perfect overlap, the optimistic latency floor is:

$$T_{\mathrm{ideal}} = \max(T_{\mathrm{gpu}}, T_{\mathrm{cpu}}, T_{\mathrm{um}}, T_{\mathrm{nvme}}, T_{\mathrm{intra}}, T_{\mathrm{inter}}, T_{\mathrm{serial}}).$$

With no overlap:

$$T_{\mathrm{no\ overlap}} = T_{\mathrm{gpu}} + T_{\mathrm{cpu}} + T_{\mathrm{um}} + T_{\mathrm{nvme}} + T_{\mathrm{intra}} + T_{\mathrm{inter}} + T_{\mathrm{serial}}.$$

Real execution lies between these models and may be slower because of queueing, dependencies,
launch overhead, cache misses, and contention. A roofline must never be labeled as measured
throughput.

## 5. GB10 shared-memory constraint

On one NVIDIA GB10, CPU, GPU, CUDA copies, pinned staging, and NVMe DMA share the same coherent
LPDDR system. They cannot each claim the full vendor bandwidth independently.

A useful accounting identity is:

$$B_{\mathrm{um}} = B_{\mathrm{dense}} + B_{\mathrm{active\ expert}} + B_{\mathrm{KV}} + B_{\mathrm{head}} + B_{\mathrm{NVMe\ destination}} + B_{\mathrm{staging/frame}} + B_{\mathrm{temporary}}.$$

The advertised accelerator peak is not a sustained batch-1 MoE rate. Ferrule requires measured
kernel and memory efficiencies for the exact shape and data type.

The historical read-only storage characterization found an observed `io_uring`/registered-pinned
ceiling of approximately `10.53 GiB/s` at queue depth 2. This number is a platform measurement,
not a guarantee for expert reads under concurrent GPU and memory traffic.

## 6. Routed-expert I/O budget

For the current profile:

- routed layers: 43;
- selected experts per row and layer: 6;
- one expert payload: approximately `12.75 MiB`.

The all-cold payload for one row is:

$$B_{\mathrm{routed,cold}} = 43 \cdot 6 \cdot 12.75\ \mathrm{MiB} \approx 3.21\ \mathrm{GiB/pass}.$$

At an observed storage ceiling of `10.53 GiB/s`, the storage-only all-cold roof is:

$$R_{\mathrm{nvme,cold}} \le \frac{10.53}{3.21} \approx 3.28\ \mathrm{passes/s}.$$

This is a **model**, not end-to-end throughput.

For $Q$ verification rows, let $E_{\ell,q}$ be the exact selected-expert set for row $q$ at layer
$\ell$, and let $S_e$ be one expert payload. The all-cold route union is:

$$B_{\mathrm{routed,cold}}(Q) = S_e \sum_{\ell=1}^{43}\left|\bigcup_{q=1}^{Q}E_{\ell,q}\right|.$$

Therefore:

$$3.21\ \mathrm{GiB} \le B_{\mathrm{routed,cold}}(Q) \le 3.21Q\ \mathrm{GiB},$$

until each layer's union reaches its expert count. The lower bound requires perfect route reuse;
the upper bound represents disjoint selected sets. Real route traces, not an independence
assumption, must supply the release calculation.

For desired output throughput $R^*$, mean committed tokens $C$, and observed storage bandwidth
$W_{\mathrm{nvme}}$, the per-cycle read budget is:

$$B^*_{\mathrm{NVMe/cycle}} \le \frac{C W_{\mathrm{nvme}}}{R^*}.$$

At $C=4$, $R^*=16\ \mathrm{tok/s}$, and $W_{\mathrm{nvme}}=10.53\ \mathrm{GiB/s}$:

$$B^*_{\mathrm{NVMe/cycle}} \le \frac{4\cdot10.53}{16} \approx 2.63\ \mathrm{GiB/cycle}.$$

The target therefore requires a cache-heavy regime with useful route reuse and proven overlap.
Increasing queue depth or issuing more speculative I/O does not change this byte budget.

## 7. Overlap and uncovered critical path

Concurrency is not proof of overlap. Ferrule only counts overlap when a timeline establishes that:

1. transaction A is blocked on a real read, upload, CUDA event, or publish dependency;
2. transaction B performs useful ready work during the same interval;
3. both operations retain valid hard-resource custody and transaction ownership;
4. completion and wake timestamps can be reconciled with the registry ledger.

For a wait interval $W$ and useful runnable work union $U$ during that interval:

$$T_{\mathrm{uncovered}} = |W \setminus U|.$$

Shared waits are unioned, not multiplied by waiter count. Queueing caused by exhausted hard credits
must be reported separately from physical read or upload latency.

Required stage accounting includes:

- reserve queue and fair-dispatch wait;
- read submission, bytes, CQE, and pinned-ready time;
- upload submission, bytes, CUDA event, and frame-ready time;
- install activation, publish, and targeted wake;
- continuation ready, resume, finish, and detach;
- uncovered read/upload/wake/resume duration.

## 8. Hard-resource evidence

A valid run reports capacity, high-water mark, acquisition wait, release, and shutdown residual for
every configured hard resource. The current scheduler models 13 resource classes, including:

- SQE;
- registered pinned slab;
- read bytes;
- upload slot;
- upload bytes;
- expert frame;
- selected lease;
- load operation;
- execution arena;
- KV state;
- continuation;
- ready cohort;
- transaction-scoped execution ownership.

Names in serialized output are schema-controlled; use the emitted schema rather than reconstructing
counter names from this prose. Every high-water mark must remain at or below capacity, and all
shutdown residual grants must be zero.

## 9. Reproducible commands

### Build

```bash
just build-cuda sm_121a
```

### Direct resident-driver probe

```bash
just dsv4-runtime-driver-bench \
  "Hello" \
  "Explain Ferrule in one sentence." \
  8 \
  2 \
  4096 \
  43
```

The underlying command supports JSON artifacts and additional CLI arguments through the `just`
recipe. Freeze all non-default arguments in the report.

### OpenAI-compatible serving sweep

Terminal 1:

```bash
just dsv4-serve
```

Terminal 2:

```bash
just dsv4-vllm-bench sweep
```

For the current acceptance pass, preserve results for `c1`, `c2`, and `c4` and report request
success/failure alongside throughput. The exact workload parameters and raw output directory must
be included with the result.

### Validation lanes

```bash
just test-runtime
just test-model
just test-cuda-required
just test-cutlass-provider sm_121a
```

Do not replace the CUDA build with a plain feature-enabled Cargo invocation when the repository's
CUDA toolchain requires `cargo-oxide` and an explicit architecture.

## 10. Artifact checklist

Keep these artifacts together for each accepted run:

- command line and environment overrides;
- git revision and dirty-state report;
- stdout JSON and stderr log;
- wall-time record and process exit code;
- device, driver, CUDA, filesystem, and storage identity;
- runtime materialization and driver statistics;
- operator counters and acceptance distribution;
- stage bytes/count/time and uncovered critical path;
- hard-resource capacity/high-water/residual table;
- request-level success/failure output;
- profiler or timeline needed to prove overlap.

A run that times out, emits a protocol failure, leaves resource ownership at shutdown, or lacks
externally committed token accounting is diagnostic evidence, not an accepted throughput result.

## 11. Common reporting errors

Do not:

- present `16 tok/s` as achieved before the frozen acceptance run passes;
- convert a single successful `n1` completion into a throughput claim;
- report internal cohort-cycle rate as externally committed output throughput;
- infer overlap from concurrency, GPU utilization, or lower wall time alone;
- multiply a single-row rate by accepted length without width-dependent work and route bytes;
- divide logical requested bytes by time when physical single-flight bytes differ;
- omit failed or timed-out requests;
- use vendor peak FLOP/s as a measured small-batch rate;
- extrapolate one GB10 result to another accelerator profile;
- describe a historical build or benchmark as evidence for the current source revision.

The roadmap may close a performance gate only when correctness, lifecycle, resource, and
reproducibility evidence all refer to the same source revision and workload.
