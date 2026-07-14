<h1 align="center">Ferrule</h1>

<p align="center">
  <strong>Rust-native LLM runtime for sparse MoE, expert streaming, and hardware-aware execution.</strong>
</p>

<p align="center">
  Router decisions, selected experts, quantized weights, KV cache, expert residency,
  and storage objects are explicit runtime concepts — not hidden behind opaque kernels.
</p>

<p align="center">
  <img alt="Rust" src="https://img.shields.io/badge/Rust-native-f97316?style=flat-square" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-cuda--oxide-22c55e?style=flat-square" />
  <img alt="MoE" src="https://img.shields.io/badge/MoE-router%20%2B%20top--k%20experts-8b5cf6?style=flat-square" />
  <img alt="WeightPack" src="https://img.shields.io/badge/WeightPack-execution%20artifact-2563eb?style=flat-square" />
  <img alt="Storage" src="https://img.shields.io/badge/Storage-residency%20vocabulary-06b6d4?style=flat-square" />
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#current-milestone">Milestone</a> ·
  <a href="#core-features">Features</a> ·
  <a href="#dgx-spark-platform-evidence-and-sota-direction">GB10 evidence</a> ·
  <a href="docs/ferrule_arch.md">Architecture</a> ·
  <a href="docs/execution-engine-architecture.md">Execution Engine</a> ·
  <a href="docs/ROADMAP.md">Roadmap</a> ·
  <a href="docs/storage-residency-architecture.md">Storage</a>
</p>

---

## Current milestone

Ferrule has a real, lossless 43-layer CUDA DeepSeek-V4 Flash target path on GB10.
Native packed decode, ragged prefill, mixed batches, physical paged multi-plane KV,
FP4 routed MoE, compressed attention state, and deterministic greedy generation are
active. DSpark metadata is represented, but proposal/verification/rollback execution
is not implemented yet.

The completed execution foundation includes:

- dependency-neutral `ExecutionBatch`, prepared plans, and persistent arenas;
- explicit per-sequence state and native packed multi-session execution;
- runtime-owned paged KV transactions, rollback, COW, and prefix sharing;
- device routing and stable expert slot/generation/lease residency;
- production `O_DIRECT + io_uring` expert reads into registered CUDA-pinned slabs;
- bounded slab admission, selected takeover, and ordinary device expert frames;
- compact router handoff, device-global output-head top-k merge, and zero
  whole-stream synchronization in steady decode;
- OpenAI-compatible HTTP/SSE serving with bounded admission and cancellation;
- reusable `vllm bench serve` compatibility and concurrency workflows.

Recent correctness work also made packed concurrency safe when a prefetched expert's
reserved victim is selected by the same batch: Ferrule cancels only the conflicting
reservation, preserves completed staging/upload work, and re-reserves a safe slot.

Ferrule is **not yet a single-node SOTA claim**. The remaining architecture work is:

1. establish an exact resident/no-I/O 43-layer target-pass roofline;
2. add a device all-hit path so host work is proportional to actual expert misses;
3. fuse the GB10 FP4 routed-MoE path and add stable graph buckets;
4. schedule continuous batches by incremental unique-expert bytes, not row count alone;
5. implement exact DSpark proposal, packed verification, commit, and rollback.

The end target is **15–17 accepted output tokens/s**, not 15–17 complete target-model
passes/s. See the [execution roadmap](docs/ROADMAP.md) for measured gates and phase
ordering.

---

## DGX Spark platform evidence and SOTA direction

The following measurements were collected on **2026-07-14** on one DGX Spark with
an integrated NVIDIA GB10 GPU. They establish the platform-specific I/O and memory
constraints for Ferrule's single-node DeepSeek-V4 work; they are not yet an
end-to-end model throughput claim.

### Test platform

```text
Superchip             NVIDIA GB10 Grace Blackwell
CPU                   20-core Arm, sharing the same coherent LPDDR5x
GPU                   compute capability 12.1 / sm_121a
Vendor FP4 AI peak    up to 1 PFLOP; not a batch-1 GEMV sustained rate
Unified memory        127.6 GB available, 128 GB nominal
Vendor memory BW      273 GB/s LPDDR5x, shared by CPU, GPU, copies, and I/O staging
CUDA / driver         CUDA 13.0 / 580.82.09
Kernel                Linux 6.11.0-1014-nvidia, aarch64
Storage               Samsung MZALC4T0HBL1-00B07 3.7 TB NVMe
Filesystem            ext4
Checkpoint            DeepSeek-V4-Flash-DSpark, 48 shards, approximately 156 GiB
Routed experts        approximately 137.1 GiB
GDS components        libnvidia-fs 2.20.5 and libcufile 1.15.1.6 installed
```

The capacity and vendor peak figures follow the
[NVIDIA DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/).
The checkpoint is larger than physical memory before accounting for KV, dense
weights, runtime arenas, pinned buffers, and the OS. This is therefore a deadline
streaming problem rather than a conventional "load the model once" deployment.

### Capacity and throughput roofline

The following equations are **upper-bound models**, not Ferrule benchmark results.
They follow the compute/I/O latency-model style used by
[MoE-SpeQ](papers/2511.14102-moe-speq.pdf),
[SP-MoE](papers/2510.10302-sp-moe.pdf),
[Klotski](papers/2502.06888-klotski.pdf), and
[Fiddler](papers/2402.07033-fiddler.pdf).

For one exact target pass, define:

| Symbol | Meaning |
|---|---|
| $F_{\mathrm{gpu}}, F_{\mathrm{cpu}}$ | GPU and CPU operations executed by the pass |
| $B_{\mathrm{um}}$ | total bytes crossing the shared LPDDR memory system |
| $B_{\mathrm{nvme}}$ | bytes that must be read from NVMe |
| $B_{\mathrm{intra}}, B_{\mathrm{inter}}$ | bytes communicated over intra-node and inter-node fabrics |
| $P_r$ | peak compute rate of resource $r$ |
| $W_r$ | peak bandwidth of resource $r$ |
| $\eta_r \in (0, 1]$ | measured efficiency of resource $r$ |
| $C_r$ | number of latency-sensitive communication rounds |
| $L_r$ | latency per communication round |
| $T_{\mathrm{serial}}$ | unavoidable sequential/control latency |

Per-resource lower bounds are:

$$
\begin{aligned}
T_{\mathrm{gpu}}   &= \frac{F_{\mathrm{gpu}}}{\eta_{\mathrm{gpu}} P_{\mathrm{gpu}}}, \\
T_{\mathrm{cpu}}   &= \frac{F_{\mathrm{cpu}}}{\eta_{\mathrm{cpu}} P_{\mathrm{cpu}}}, \\
T_{\mathrm{um}}    &= \frac{B_{\mathrm{um}}}{\eta_{\mathrm{um}} W_{\mathrm{um}}}, \\
T_{\mathrm{nvme}}  &= \frac{B_{\mathrm{nvme}}}{\eta_{\mathrm{nvme}} W_{\mathrm{nvme}}}, \\
T_{\mathrm{intra}} &= \frac{B_{\mathrm{intra}}}{\eta_{\mathrm{intra}} W_{\mathrm{intra}}}
                      + C_{\mathrm{intra}} L_{\mathrm{intra}}, \\
T_{\mathrm{inter}} &= \frac{B_{\mathrm{inter}}}{\eta_{\mathrm{inter}} W_{\mathrm{inter}}}
                      + C_{\mathrm{inter}} L_{\mathrm{inter}}.
\end{aligned}
$$

With perfect overlap, the optimistic roofline is the slowest resource:

$$
T_{\mathrm{ideal}} = \max\!\left(
T_{\mathrm{gpu}},
T_{\mathrm{cpu}},
T_{\mathrm{um}},
T_{\mathrm{nvme}},
T_{\mathrm{intra}},
T_{\mathrm{inter}},
T_{\mathrm{serial}}
\right),
\qquad
R_{\mathrm{target}} \le \frac{1}{T_{\mathrm{ideal}}}.
$$

With no overlap, the corresponding latency model is:

$$
T_{\mathrm{no\text{-}overlap}}
= T_{\mathrm{gpu}} + T_{\mathrm{cpu}} + T_{\mathrm{um}} + T_{\mathrm{nvme}}
+ T_{\mathrm{intra}} + T_{\mathrm{inter}} + T_{\mathrm{serial}}.
$$

Real execution is always slower than the optimistic roof because dependencies, kernel
launches, queueing, cache misses, and resource contention reduce $\eta_r$ and limit
overlap.

#### One DGX Spark

CPU, GPU, CUDA copies, pinned staging, and NVMe DMA all share the same LPDDR system.
They must not be modeled as independent $273\ \mathrm{GB/s}$ domains:

$$
\begin{aligned}
B_{\mathrm{um}} ={}& B_{\mathrm{dense}}
+ B_{\mathrm{active\text{-}expert}}
+ B_{\mathrm{KV}}
+ B_{\mathrm{head}} \\
&+ B_{\mathrm{nvme\text{-}destination}}
+ B_{\mathrm{staging\leftrightarrow frame}}
+ B_{\mathrm{temporary}}.
\end{aligned}
$$

The GPU and CPU ceilings are:

$$
\begin{aligned}
R_{\mathrm{gpu}}
&\le \min\!\left(
\frac{\eta_{\mathrm{gpu}} \cdot 1\ \mathrm{PFLOP/s}}{F_{\mathrm{gpu}}},
\frac{\eta_{\mathrm{um}} \cdot 273\ \mathrm{GB/s}}{B_{\mathrm{gpu}}}
\right), \\
R_{\mathrm{cpu}}
&\le \min\!\left(
\frac{\eta_{\mathrm{cpu}} P_{\mathrm{cpu}}}{F_{\mathrm{cpu}}},
\frac{\eta_{\mathrm{um}} \cdot 273\ \mathrm{GB/s}}{B_{\mathrm{cpu}}}
\right).
\end{aligned}
$$

The advertised FP4 peak is a very loose bound for batch-1 matrix-vector work. NVIDIA
does not publish a directly comparable sustained MoE rate for the 20-core CPU, so
$P_{\mathrm{cpu}}$ must be measured for the selected data type and kernel. CPU and GPU
cannot both claim the full $273\ \mathrm{GB/s}$ simultaneously because they share LPDDR. The current
lossless GB10 path uses the CPU primarily for control and I/O rather than expert math.
A Fiddler-style CPU expert path is useful only when:

$$
T_{\mathrm{cpu\text{-}expert}}
< T_{\mathrm{stage/copy}} + T_{\mathrm{gpu\text{-}expert}}.
$$

The model's all-cold routed-expert payload is known directly from its shape:

$$
B_{\mathrm{routed,cold}}
= 43 \cdot 6 \cdot 12.75\ \mathrm{MiB}
= 3.21\ \mathrm{GiB/target\ pass}.
$$

At the observed $10.53\ \mathrm{GiB/s}$ storage ceiling, even perfect all-cold
streaming gives:

$$
R_{\mathrm{nvme,cold}}
\le \frac{10.53\ \mathrm{GiB/s}}{3.21\ \mathrm{GiB/pass}}
\approx 3.28\ \mathrm{target\ passes/s}.
$$

That is a storage/model roofline, not an end-to-end Ferrule result. Caching and
prediction reduce $B_{\mathrm{nvme}}$; wrong prefetch increases it.

For a fully staged/resident working set, NVMe can leave the steady critical path, but
LPDDR cannot. Converting the vendor bandwidth to binary units gives:

$$
273\ \mathrm{GB/s}
\approx 254.25\ \mathrm{GiB/s}.
$$

For an illustrative exact pass moving $B_{\mathrm{um}} \in [20,22]\ \mathrm{GiB}$,
the ideal memory-only ceiling is:

$$
R_{\mathrm{um}}
\le \frac{254.25\ \mathrm{GiB/s}}{B_{\mathrm{um}}}
\in [11.6,12.7]\ \mathrm{target\ passes/s}.
$$

The exact $B_{\mathrm{um}}$ must come from the S0 target-pass roofline in
[`docs/ROADMAP.md`](docs/ROADMAP.md); 20–22 GiB is an intentionally optimistic
capacity example, not a measured Ferrule token rate. It illustrates why 15–17 full
43-layer passes/s is not the correct single-Spark target.

For DSpark, let $A$ be the average committed tokens per exact target verification
cycle. The accepted-token roof is:

$$
R_{\mathrm{accepted}} \le \frac{A}{T_{\mathrm{cycle}}},
$$

where

$$
T_{\mathrm{cycle}}
= T_{\mathrm{draft}}
+ T_{\mathrm{verify}}
+ T_{\mathrm{commit/rollback}}
- T_{\mathrm{safe\ overlap}}.
$$

For example, $A=4$ accepted tokens from a $T_{\mathrm{cycle}}=250\ \mathrm{ms}$
complete cycle imply

$$
R_{\mathrm{accepted}} \le \frac{4}{0.25\ \mathrm{s}} = 16\ \mathrm{tokens/s}.
$$

This is the architectural meaning of Ferrule's 15–17 tok/s goal; the numbers are a
target operating point, not a current result.

Combining the storage and resident-memory roofs gives the following intentionally
optimistic single-Spark envelope:

| Average accepted tokens $A$ | All-cold NVMe roof | Resident LPDDR roof |
|---:|---:|---:|
| $1$ | $3.28$ accepted tok/s | $11.6$–$12.7$ accepted tok/s |
| $2$ | $6.56$ accepted tok/s | $23.2$–$25.4$ accepted tok/s |
| $4$ | $13.12$ accepted tok/s | $46.4$–$50.8$ accepted tok/s |

The table uses:

$$
R_{\mathrm{accepted,ideal}}(A)
\approx A \cdot R_{\mathrm{target,ideal}}.
$$

It assumes multi-token verification does not increase active expert bytes, draft and
rollback are free or hidden, acceptance is exactly $A$, and resource efficiencies are
already represented by the selected roof. Real throughput is lower.

For a desired accepted throughput $R^*_{\mathrm{accepted}}$, the per-cycle storage
budget is:

$$
B^*_{\mathrm{nvme,cycle}}
\le \frac{A W_{\mathrm{nvme}}}{R^*_{\mathrm{accepted}}}.
$$

At $A=4$, $R^*_{\mathrm{accepted}}=16\ \mathrm{tok/s}$, and
$W_{\mathrm{nvme}}=10.53\ \mathrm{GiB/s}$:

$$
B^*_{\mathrm{nvme,cycle}}
\le \frac{4 \cdot 10.53}{16}
\approx 2.63\ \mathrm{GiB/cycle}.
$$

Thus 15–17 tok/s is physically plausible only in the cached/speculative regime, not
when every verification cycle reloads the full $3.21\ \mathrm{GiB}$ routed set.

#### One host with multiple GPUs

A DGX Spark contains one GB10; this subsection describes Ferrule's extension to a
separate multi-GPU host rather than multiple internal GPUs in one Spark. For $G$ GPUs
in one host, the ideal node roofline becomes:

$$
T_{\mathrm{node}} = \max\!\left(
\frac{F_{\mathrm{gpu}}}{G\eta_{\mathrm{gpu}}P_{\mathrm{gpu}}},
\frac{B_{\mathrm{device}}}{G\eta_{\mathrm{mem}}W_{\mathrm{device}}},
\frac{B_{\mathrm{host}}}{\eta_{\mathrm{host}}W_{\mathrm{host}}},
\frac{B_{\mathrm{nvme}}}{\eta_{\mathrm{storage}}W_{\mathrm{storage,agg}}},
\frac{B_{\mathrm{intra}}}{\eta_{\mathrm{intra}}W_{\mathrm{intra}}}
+ C_{\mathrm{intra}}L_{\mathrm{intra}},
T_{\mathrm{serial}}
\right),
\qquad
R_{\mathrm{node}} \le \frac{1}{T_{\mathrm{node}}}.
$$

Capacity and bandwidth scale differently:

$$
M_{\mathrm{resident}}
\le \sum_{g=1}^{G} M_g + M_{\mathrm{host,budget}},
$$

$$
W_{\mathrm{storage,agg}}
\le \sum_{d \in \mathrm{independent\ paths}} W_d,
\qquad
P_{\mathrm{node}}
\le \sum_{g=1}^{G} P_g
\quad \text{only for partitionable work}.
$$

If aggregate GPU memory holds the checkpoint plus KV/arenas, steady expert NVMe traffic
can approach zero. If the model still streams from one drive, adding GPUs does not
remove the storage roof. Tensor parallelism introduces collectives; expert parallelism
reduces weight duplication but introduces dispatch/combine traffic. The scheduler must
choose TP/EP/PP from measured communication and memory costs rather than GPU count.

The ideal linear speedup and parallel efficiency are:

$$
S(G) = \frac{R_G}{R_1} \le G,
\qquad
E(G) = \frac{S(G)}{G} \le 1.
$$

#### Multiple nodes with multiple GPUs

For $N$ nodes with $G$ GPUs each:

$$
T_{\mathrm{cluster}} = \max\!\left(
\frac{F_{\mathrm{gpu}}}{NG\eta_{\mathrm{gpu}}P_{\mathrm{gpu}}},
\frac{B_{\mathrm{device}}}{NG\eta_{\mathrm{mem}}W_{\mathrm{device}}},
\frac{B_{\mathrm{storage}}}{\eta_{\mathrm{storage}}W_{\mathrm{storage,cluster}}},
\frac{B_{\mathrm{intra}}}{\eta_{\mathrm{intra}}W_{\mathrm{intra}}}
+ C_{\mathrm{intra}}L_{\mathrm{intra}},
\frac{B_{\mathrm{inter}}}{\eta_{\mathrm{inter}}W_{\mathrm{network}}}
+ C_{\mathrm{inter}}L_{\mathrm{network}},
T_{\mathrm{serial}}
\right),
\qquad
R_{\mathrm{cluster}} \le \frac{1}{T_{\mathrm{cluster}}}.
$$

For expert-parallel decode, a useful first-order activation-traffic estimate is:

$$
B_{\mathrm{EP}}
\approx 2p_{\mathrm{remote}}Lkbh s,
$$

where $L$ is the layer count, $k$ is top-k, $b$ is the batch row count, $h$ is hidden
size, and $s$ is bytes per activation element. The factor two covers dispatch and
returned expert outputs. With $L=43$, $k=6$, $b=1$, $h=4096$, $s=2$ bytes, and
$p_{\mathrm{remote}}=1$:

$$
B_{\mathrm{EP}}
\approx 2 \cdot 1 \cdot 43 \cdot 6 \cdot 1 \cdot 4096 \cdot 2
= 4{,}227{,}072\ \mathrm{bytes}
\approx 4.03\ \mathrm{MiB/target\ pass}.
$$

This excludes router metadata and collective overhead. The byte count
is manageable on a fast fabric, but up to 43 latency-sensitive routing rounds can
still dominate small-batch decode. Cross-node tensor parallelism similarly pays
collective latency at many sublayers.

For a Spark cluster, a $200\ \mathrm{Gb/s}$ network path has the vendor-level
unidirectional bandwidth ceiling:

$$
W_{\mathrm{network}}
\le \frac{200\ \mathrm{Gb/s}}{8}
= 25\ \mathrm{GB/s}
\approx 23.3\ \mathrm{GiB/s}.
$$

Sending the $3.21\ \mathrm{GiB}$ all-cold routed weight set over that path every pass
would cap the weight-transfer roof at:

$$
R_{\mathrm{remote\text{-}weight}}
\le \frac{23.3\ \mathrm{GiB/s}}{3.21\ \mathrm{GiB/pass}}
\approx 7.3\ \mathrm{target\ passes/s}.
$$

Expert-parallel placement should therefore keep weights local to their owner and
exchange the MiB-scale activations instead. With independent local NVMe devices and
balanced sharding, the ideal storage ceiling is bounded by:

$$
W_{\mathrm{storage,cluster}}
\le \sum_{n=1}^{N} W_{\mathrm{nvme},n}
\approx N W_{\mathrm{nvme}}
\quad \text{only for independent, balanced paths}.
$$

One shared storage path does not scale with node count.

Practical scaling order is therefore:

1. use aggregate accelerator memory to eliminate steady weight streaming;
2. prefer expert/pipeline placement that minimizes per-layer cross-node collectives;
3. increase useful batch or verification width to amortize communication latency;
4. replicate storage or shard it across independent paths when streaming remains;
5. report scaling efficiency and accepted tokens/s, not only aggregate peak FLOPs.

### GDS capability result

`gdscheck -p` passed the general platform checks but reported both of the constraints
that matter for the actual data path:

```text
GPU NVIDIA GB10: Model Not Supported
NVMe P2PDMA: Unsupported
use_compat_mode: true
```

File registration succeeded, but device-buffer registration and read verification
failed with `nvfs internal driver error`. `gdsio -x 0` with explicit buffer
registration failed in the same way. Consequently, Ferrule must not describe the
GB10 path as direct NVMe-to-GPU GDS merely because `nvidia_fs` and `libcufile` are
installed. cuFile compatibility mode remains a useful baseline, not the preferred
production architecture.

### Read-only storage A/B

The storage runs read existing safetensors shards without modifying them. Each run
read 1 GiB in 16 MiB requests with one worker unless the queue depth says otherwise;
different shards were used to reduce cross-run cache contamination. The explicitly
cold page-cache run used `POSIX_FADV_DONTNEED`.

| Data path | Throughput | Average request latency | Result |
|---|---:|---:|---|
| cuFile `x0`, registered device buffer | — | — | Failed: `nvfs internal driver error` |
| cuFile `x0 -b`, compatibility mode | 6.79 GiB/s | 2.03 ms | Passed, not direct GDS |
| `gdsio` CPU→GPU `x2` | 6.13 GiB/s | 2.37 ms | Passed |
| page cache→GPU, nominal cold run | 1.35 GiB/s | 11.14 ms | Passed |
| page cache→GPU, warm | 5.39 GiB/s | 2.84 ms | Passed |
| page cache, cold after `DONTNEED` | 1.05 GiB/s | 14.47 ms | Passed |
| raw `io_uring` + registered pinned buffers, QD1 | 6.92 GiB/s | 1.61 ms | Passed |
| raw `io_uring` + registered pinned buffers, QD2 | **10.53 GiB/s** | **2.21 ms** | Passed |
| raw `io_uring` + registered pinned buffers, QD4 | 10.71 GiB/s | 5.07 ms | Passed |
| raw `io_uring` + registered pinned buffers, QD8 | 10.70 GiB/s | 10.51 ms | Passed |

QD2 reaches nearly all observed NVMe throughput while avoiding the rapidly growing
latency at QD4/QD8. The production reader therefore uses
`O_DIRECT + io_uring fixed files + registered pinned buffers` with QD≈2, then tunes
from actual expert deadlines rather than maximizing queue depth. The one-off A/B
harness was removed after the result was recorded; production counters and reusable
model benchmarks remain in the runtime and `justfile`.

### What unified memory changes—and what it does not

GB10 has one coherent LPDDR memory system, so an H2D/D2H operation is not a transfer
across a discrete-GPU PCIe link. CUDA allocations nevertheless remain logically
separate. Copying from a `cudaHostAlloc` staging allocation into a `cudaMalloc`
allocation still reads and writes LPDDR, retains two copies while both are alive,
consumes capacity, and introduces stream/event dependencies. Coherence removes the
physical PCIe boundary; it does not make copies free.

The following test measured first GPU consumption of a staged payload. It used a
memory-consumption kernel, **not** Ferrule's FP4 expert Tensor Core kernel.

| 16 MiB path | First-consumption latency | Effective payload bandwidth |
|---|---:|---:|
| pinned→device copy + first kernel | 635 µs | 24.6 GiB/s |
| pinned→device copy + warm kernel | 400 µs | 39.0 GiB/s |
| device allocation, kernel only | 123 µs | 127.1 GiB/s |
| mapped pinned, first kernel | **150 µs** | **104.0 GiB/s** |
| mapped pinned, warm kernel | 148 µs | 105.3 GiB/s |
| managed, first kernel | 171 µs | 91.5 GiB/s |
| managed, warm kernel | 192 µs | 81.5 GiB/s |

A second run used 13 MiB, close to one 12.75 MiB routed-expert payload:

| 13 MiB path | First-consumption latency | Effective payload bandwidth |
|---|---:|---:|
| pinned→device copy + first kernel | 521 µs | 24.4 GiB/s |
| device allocation, kernel only | 123 µs | 103.3 GiB/s |
| mapped pinned, first kernel | **139 µs** | **91.3 GiB/s** |
| mapped pinned, warm kernel | 137 µs | 92.4 GiB/s |
| managed, first kernel | 155 µs | 81.8 GiB/s |
| managed prefetch + first kernel | 891 µs | 14.2 GiB/s |

On this platform, direct mapped-pinned consumption avoided roughly 380–485 µs of
copy-path latency per expert-sized payload while remaining only 16–27 µs slower than
the device-allocation kernel-only result. Managed prefetch was substantially worse in
the 13 MiB run. The one-off generic memory A/B harness was removed after recording
these results.

A one-off follow-up then ran the **real routed FP4 Tensor Core path** with
`layer=0/expert=0`, the same 12.75 MiB checkpoint payload and input, 10 warmup runs,
and 50 measured runs per backing:

| Real FP4 backing | Build/stage | First use | Warm steady execution |
|---|---:|---:|---:|
| device frame | 2.213 ms | 1,581.2 µs | **518.2 µs** |
| mapped pinned | 4.375 ms | **1,395.8 µs** | 1,155.9 µs |

The outputs were bitwise identical (`max_abs_diff = 0.0`), proving that the existing
stable-slot FP4 kernels can dereference mapped system-memory pointers without changing
model math. However, mapped-pinned steady execution was about **2.23× slower** than a
device frame. The generic bandwidth microbenchmark therefore does not predict actual
Tensor Core expert throughput.

This changes the production decision: mapped/registered pinned memory is an I/O staging
tier, not the default hot-expert compute tier. High-confidence and selected experts
should be promoted to an ordinary device frame before their deadline. Direct execution
from mapped memory may remain an emergency late-miss option only if end-to-end profiling
shows it beats waiting for the copy. The one-off harness was removed after recording
this result; it was not added to the permanent test surface.

### Literature-derived runtime decision

The papers retained under [`papers/`](papers/) point to a common lossless design:

1. **Use two prediction horizons.** Long-distance history/semantic signals may stage
   coarse expert pages; current-token partial routing trajectories make the near-layer
   prediction used for urgent promotion.
2. **Make I/O deadline-aware.** Admit work by target layer, measured read/pin latency,
   and confidence. A continuous bounded queue is preferable to submitting fixed-K
   work for all 43 layers.
3. **Keep speculation semantically invisible.** The real router always determines
   expert IDs and weights. A bad prediction may waste I/O or cause a wait, but it must
   never substitute an expert or change the generated token.
4. **Separate host staging from device-slot reservation.** Far-ahead reads must not
   occupy stable expert slots. Reserve/publish only when an expert is near its
   consumption deadline.
5. **Use a cutoff and measured cache policy.** Published results warn that excessive
   depth can cause cache thrashing; depth around five layers and resident capacity near
   twice the active-expert count are starting hypotheses, not constants. DeepSeek-V4
   top-6 traces must determine the actual values.
6. **Do not use an unscored catalog fallback.** Wrong prefetch can be slower than no
   prefetch. The runtime should stage fewer experts when confidence is low.

For GB10, this becomes the following proposed pipeline:

```text
far, lower-confidence prediction
  -> 16 MiB-class O_DIRECT/io_uring StageToRegisteredPinned
  -> QD approximately 2, deadline/cutoff admission, no stable-slot reservation

near, current-token cross-layer trajectory
  -> high-confidence L+1 promotion
  -> async pinned-to-device-frame copy before the target-layer deadline
  -> publish the stable slot only when the device frame is ready

real router result
  -> authoritative selected experts and weights
  -> promote/wait on a miss; never change model semantics
```

The production reader, bounded selected takeover, compact router handoff, and
device-global output-head merge are now implemented. The next optimization order is:

1. measure an exact resident/no-I/O target-pass roofline;
2. execute all-hit routed layers without host materialization;
3. fuse the profiler-proven GB10 FP4 MoE stages and capture stable row buckets;
4. drive residency and batch admission from route traces, deadlines, and incremental
   unique-expert bytes;
5. implement DSpark proposal, exact packed verification, and rollback with one
   acceptance-aware I/O governor.

Ferrule cannot yet claim single-node SOTA: that requires reproducible end-to-end warm
ITL, TTFT, throughput, memory, quality, and DSpark acceptance results against competing
runtimes under the same model, prompt distribution, tokenizer, sampling, and
concurrency. Current end-to-end diagnostic numbers are intentionally not presented in
this README; the roadmap defines the benchmark gates and raw artifacts remain under
`target/bench/`.

---

## Core features

| Feature | Why it matters |
|---|---|
| **Policy-composed model descriptions** | Model families map source tensors into semantic attention, FFN/MoE, KV, residency, quantization, and speculation policies. DSV4 now separates fully prepared layers, backend expert runtime, explicit sequence state, and reusable eager arenas. |
| **Neutral execution ABI** | One public `ExecutionBatch` expresses packed/ragged rows, phases, state slots, KV bindings, and strict logits intent; runtime correlation stays private and native multi-session execution consumes it directly. |
| **MoE-first execution** | Router logits, hash/top-k selection, selected experts, shared experts, and DSV4 expert-streaming mechanisms are explicit objects rather than opaque kernel details. Runtime owns cross-request stable-slot/generation/lease residency. |
| **Storage/residency and memory vocabulary** | `StorageObjectId`, `ObjectLocator`, `Placement`, and `ObjectReplica` describe storage; generic `MemoryPoolLimits`, `MemoryPoolStats`, and `OwnerMemoryLru` enforce local whole-expert host/pinned retention without model-specific policy leakage. |
| **cuda-oxide kernels** | Custom CUDA kernels integrated with the Rust runtime: quantized GEMV, packed FP4 expert execution, sparse attention, artifact-preserving operators. |
| **Safetensors source binding** | Inspect and bind Hugging Face safetensors by semantic role, with bounded reads instead of loading a 100 GB+ checkpoint into RAM. |
| **WeightPack execution artifact** | Layer weights quantized once and reloaded from a Ferrule-owned package. GGUF remains a compatibility/PK path. |
| **Runtime graph IR** | Opaque graph with semantic ops, typed artifact bindings, shape registry, and backend object store. Model-family names stay out of graph nodes. |
| **CPU/reference anchors** | CPU reference pieces validate CUDA kernels, source-format decoders, router behavior, and HC math — without a legacy full-model CPU runner. |
| **Edge/hardware direction** | Expert placement, streaming, WeightPack layout, and scheduling adapt to VRAM, DRAM, NVMe, and future multi-GPU / multi-node cooperation. |

---

## System vision

Ferrule is designed around a simple idea: future LLM systems need to co-design
model structure, runtime state, and hardware placement.

<p align="center">
  <img src="docs/assets/ferrule-current-architecture.svg" alt="Ferrule architecture" width="100%" />
</p>

Near term: llama.cpp-level local usability with a more explicit runtime
architecture — fast cached startup, sampling controls, templates, quality checks,
OpenAI-compatible local serving, official benchmarks, and source-preserving bring-up
for mainstream model families.

Long term: a runtime substrate for edge-cloud LLM systems:

- cloud builds model versions, WeightPack artifacts, calibration data, adapters
- edge devices run private low-latency inference and collect rollout traces
- router statistics guide expert placement, prefetch, and offload
- KV/session state becomes movable and eventually distributed
- speculation modules (DSpark/MTP) attach through a target/draft policy
- DP/TP/EP/SP/CP/PP placement evolves under one state-aware runtime

---

## Quick start

The fastest path is through `justfile` wrappers. For CUDA, prefer `just` /
`cargo oxide` commands; plain `cargo test -p ferrule-cuda` can miss cuda-oxide
artifact wiring.

1. **Check the environment:**

```bash
just cuda-info
just oxide-doctor
```

2. **Build the CUDA release binary:**

```bash
just build-cuda

# Override architecture if auto-detection fails:
FERRULE_CUDA_ARCH=sm_121a just build-cuda
```

3. **Put the local DSV4 source checkout here:**

```
models/DeepSeek-V4-Flash-DSpark
```

4. **Run real local DeepSeek V4 Flash + DSpark CUDA chat:**

```bash
just dsv4-chat tokens=128
```

Inside chat:

```
/reset    clear session state
/stats    show session stats
/experts  show DSV4 layer/cache stats
/ctx      show context window usage
/exit     quit
```

5. **One-shot DSV4 CUDA smoke:**

```bash
just dsv4-cuda-generate Hello 2 4096 --chat
```

6. **Start the OpenAI-compatible DSV4 server:**

```bash
just dsv4-serve

curl http://127.0.0.1:8000/v1/models
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"deepseek-v4","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

7. **OLMoE regression fixture (if present):**

```bash
just chat models/OLMoE-Instruct q4 -n 256
```

---

## Capability map

| Area | Status |
|---|---|
| Executable model fixture | OLMoE sparse MoE CUDA path via `GpuOlmoeRunner` |
| Real large-model milestone | DeepSeek V4 Flash exact 43-layer CUDA target path plus `bench-interactive` through `ResidentTopKDriver`; DSpark execution remains a roadmap phase. |
| Execution ABI | E1–E4 complete: public `ExecutionBatch`, prepared-executor traits, runtime-private `ScheduledBatch`, strict validation, and native packed multi-session execution |
| DSV4 execution boundary | The model owns HC/MLA/compressor/router/MoE math; runtime remains model-neutral and owns scheduling plus logical page lifecycle through generic traits |
| Expert streaming | Immutable source catalogs plus production `O_DIRECT + io_uring` fixed-file reads into registered CUDA-pinned slabs; CUDA promotes selected/on-time experts into ordinary device frames governed by stable slots, generations, leases, bounded admission, and selected takeover. |
| Storage/residency | `ferrule-runtime::storage` vocabulary plus runtime expert residency; `ferrule-common::memory` supplies model-neutral entry/byte budgets, stats, topology vocabulary, and owner-thread LRU |
| Attention | OLMoE GQA executable; DSV4 MLA sparse attention correctness-first CUDA path |
| Hyper-connections | DSV4 HC source binding + reference `hc_pre`/`hc_post`/`hc_head` |
| KV cache | E5 complete: `KvPageManager` plus bounded CUDA multi-plane pools, paged attention/indexer kernels, rollback, COW, preempt/restore, and exact-prefix sharing; host caches are reference infrastructure |
| Quantization | Q4_0/Q8_0, FP4 E2M1 + E8M0 scales, FP8 E4M3FN, mixed precision policy |
| WeightPack | mmap'd reader, streaming writer, zero-copy slices, WeightPack-only load path |
| Runtime graph | `ferrule-runtime::graph` opaque IR, `GraphProgram`, `BackendObjectStore`, and `ReferenceGraphExecutor::execute` with caller-owned `ReferenceGraphSequenceState` |
| Sampling | temperature, top-k, top-p, min-p, repeat penalty, seed, stop strings, logprobs |
| Chat templates | OLMoE, ChatML, Llama3, Qwen, DeepSeek-V4, Plain |
| Serving | `ferrule-server` + `ferrule serve`: Axum/Hyper/Tokio, dedicated model-owner thread, bounded queues, disconnect cancellation, `/health`, `/v1/models`, `/v1/chat/completions`, and `/v1/completions`; reusable `vllm bench serve` concurrency workflows; greedy only for now. |
| Structured decoding | token mask API, program-like generation API |
| Speculation | DSpark/MTP metadata is represented as a generic attachment policy; proposal, packed target verification, accepted-prefix commit, and rejected-suffix rollback remain the S6 milestone. |
| Training/RL | design target, not implemented |

---

## Useful commands

### Environment and build

```bash
just cuda-info          # Show GPU/arch detection
just oxide-doctor       # cuda-oxide environment check
just build              # Auto-detect: CUDA if available, CPU otherwise
just build-cuda         # Explicit CUDA build
just build-cpu          # CPU-only build
just check              # Quick check
```

### DeepSeek V4 / DSpark

```bash
just dsv4-chat tokens=128                        # Interactive chat
just dsv4-serve                                  # OpenAI-compatible HTTP/SSE server
just dsv4-vllm-bench smoke                       # vLLM API compatibility smoke
just dsv4-vllm-bench baseline                    # Saved official single-concurrency result
just dsv4-vllm-bench sweep                       # Saved concurrency 1/2/4 results
just dsv4-runtime-driver-bench                   # bench-interactive via ResidentTopKDriver
# positional override: prompt1 prompt2 tokens warmup chunk layers
just dsv4-runtime-driver-bench "Hello" "Explain Ferrule in one sentence." 1 0 2 43
just test-dsv4-runtime-driver-local              # opt-in ignored local runtime-driver test
just dsv4-cuda-generate Hello 2 4096 --chat       # One-shot generation
just dsv4-cuda-first-token Hello 1               # First-token diagnostic
just dsv4-cuda-probe "one two three" 3 1 0       # Layer-limited probe
just dsv4-parity-json "Who are you?" output.json # Tokenizer parity JSON
```

### Generic CLI surface

```bash
cargo run -p ferrule-cli -- info models/OLMoE-Instruct
just chat models/OLMoE-Instruct q4 -n 256        # Interactive chat wrapper
cargo run -p ferrule-cli -- cuda                 # CUDA probe + smoke benchmark
cargo run -p ferrule-cli -- inspect-weightpack path/to/model.weightpack
cargo run -p ferrule-cli -- expert-stream-smoke models/OLMoE-Instruct --layer 0 --expert 0
```

### Validation

```bash
just test           # All tests
just test-graph     # Graph IR tests
just test-runtime   # Runtime tests
just test-cuda      # CUDA tests (via cargo oxide)
just fmt            # Format check
just clippy         # Lint
just lint           # fmt + clippy
```

---

## Active development focus

1. **Run official serving smoke tests** — validate Ferrule against `vllm bench serve`
   and SGLang `benchmark.serving` using identical prompts, tokenizer, greedy settings,
   and concurrency.
2. **Build stable E7 graph buckets** — lower the unified allocation-free eager stages
   into reusable graphs without restoring the deleted token-specific one-shot path.
3. **Complete E8 device sampling and fusion** — move non-greedy sampling off the host
   path and optimize only profiler-proven stages.
4. **Connect automatic radix-prefix lookup and production metrics** — E5 already owns
   exact fork/COW primitives; independent API requests still need lookup/admission.
5. **Publish controlled comparisons** — report TTFT, TPOT, ITL, throughput, memory,
   page utilization, and quality under fixed vLLM/SGLang-compatible workloads.

---

## Documentation

| Document | Content |
|---|---|
| [Architecture](docs/ferrule_arch.md) | Repository crates, model boundaries, current runtime, serving direction, and alignment targets |
| [Execution Engine](docs/execution-engine-architecture.md) | Implemented ownership foundation: neutral ABI, prepared plans, sequence state, arenas, native batching, physical paged KV, and runtime expert residency. |
| [Serving](docs/serving.md) | Axum/Hyper/Tokio selection, dedicated model-worker ownership, OpenAI/SSE contract, cancellation, commands, cache budgets, and official benchmark targets |
| [Roadmap](docs/ROADMAP.md) | Canonical S0–S7 single-node SOTA plan: target roofline, device miss-only execution, fused kernels, trace-aware residency, expert-aware batching, and exact DSpark verification. |
| [Storage & Residency](docs/storage-residency-architecture.md) | Storage identity/placement plus implemented expert slots, leases, transfers, policy, and metrics |
| [Expert Memory & Telemetry](docs/expert-memory-architecture.md) | Model-neutral owner-thread memory pools, host/pinned expert budgets, GB10 constraints, and benchmark-safe telemetry |
| [Runtime Graph](docs/runtime-graph-architecture.md) | Device-independent graph IR, `GraphNode`, dialects, `GraphProgram`, external bindings, and explicit reference execution state |

---

## License

Apache-2.0
