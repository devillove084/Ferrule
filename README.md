<h1 align="center">Ferrule</h1>

<p align="center">
  <strong>Rust-native DeepSeek-V4 inference for one DGX Spark.</strong>
</p>

<p align="center">
  Router decisions, expert I/O, quantized weights, KV transactions, and DSpark
  verification are explicit runtime concepts — not hidden behind opaque kernels.
</p>

<p align="center">
  <a href="https://github.com/devillove084/Ferrule/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/devillove084/Ferrule/actions/workflows/ci.yml/badge.svg" /></a>
  <a href="https://codecov.io/gh/devillove084/Ferrule"><img alt="Coverage" src="https://codecov.io/gh/devillove084/Ferrule/graph/badge.svg?branch=main" /></a>
  <img alt="Rust" src="https://img.shields.io/badge/Rust-native-f97316?style=flat-square" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-cuda--oxide-22c55e?style=flat-square" />
  <img alt="MoE" src="https://img.shields.io/badge/MoE-router%20%2B%20top--k%20experts-8b5cf6?style=flat-square" />
  <img alt="WeightPack" src="https://img.shields.io/badge/WeightPack-execution%20artifact-2563eb?style=flat-square" />
  <img alt="CUTLASS" src="https://img.shields.io/badge/GEMM-CUTLASS-06b6d4?style=flat-square" />
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#current-milestone">Milestone</a> ·
  <a href="#core-features">Features</a> ·
  <a href="#dgx-spark-platform-evidence-and-sota-direction">GB10 evidence</a> ·
  <a href="docs/CUTLASS.md">CUTLASS</a> ·
  <a href="docs/serving.md">Serving</a> ·
  <a href="docs/ROADMAP.md">Roadmap</a>
</p>

---

## Current milestone

Ferrule now runs the checkpoint-native DeepSeek-V4 Flash DSpark cycle through the
production CUDA/CUTLASS and OpenAI serving path: proposal, exact target verification,
commit or rollback, correction or bonus staging, and external token reconciliation share
one runtime-owned transaction.

Production verification is a true multi-sequence cohort. Per-session proposal generation
remains isolated, ragged proposal rows are packed into one target execution, and each
sequence independently retains its exact accepted prefix. Runtime and backend KV state
are committed atomically for the cohort without replaying accepted target rows.

The completed execution foundation includes:

- one Rust-owned executable plan with prepare-time kernel-provider selection;
- a GB10/SM121a CUTLASS/CuTe provider exposing semantic target and DSpark bundles;
- arbitrary-width packed verification with per-sequence provisional checkpoints;
- runtime-owned paged KV transactions with atomic prefix retention, COW, and slot allocation;
- continuous-batching cohort formation with bounded deferral and forced progress;
- production `O_DIRECT + io_uring` expert reads into registered CUDA-pinned slabs, with no expert-streaming mmap backend;
- device routing and stable expert slot/generation/lease residency;
- owned arena checkout/checkin as the lifetime foundation for resumable execution;
- OpenAI-compatible HTTP/SSE serving validated through the official vLLM benchmark path.

Ferrule is **not yet a single-node SOTA claim**. Packed cohorts improve production
concurrency, but exact expert misses still synchronously block the model worker while
reads and uploads complete. The next milestone is resumable layer execution plus a
global I/O/CUDA completion reactor, allowing the scheduler to run other ready cohorts
while expert data moves. Official proposal parity, the all-resident kernel floor, stable
graph capture, and the frozen release suite remain required in parallel.

See the [execution roadmap](docs/ROADMAP.md) for the release contract, measured artifacts,
and phase ordering.

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

$$\begin{aligned} T_{\mathrm{gpu}}   &= \frac{F_{\mathrm{gpu}}}{\eta_{\mathrm{gpu}} P_{\mathrm{gpu}}}, \\\\ T_{\mathrm{cpu}}   &= \frac{F_{\mathrm{cpu}}}{\eta_{\mathrm{cpu}} P_{\mathrm{cpu}}}, \\\\ T_{\mathrm{um}}    &= \frac{B_{\mathrm{um}}}{\eta_{\mathrm{um}} W_{\mathrm{um}}}, \\\\ T_{\mathrm{nvme}}  &= \frac{B_{\mathrm{nvme}}}{\eta_{\mathrm{nvme}} W_{\mathrm{nvme}}}, \\\\ T_{\mathrm{intra}} &= \frac{B_{\mathrm{intra}}}{\eta_{\mathrm{intra}} W_{\mathrm{intra}}} + C_{\mathrm{intra}} L_{\mathrm{intra}}, \\\\ T_{\mathrm{inter}} &= \frac{B_{\mathrm{inter}}}{\eta_{\mathrm{inter}} W_{\mathrm{inter}}} + C_{\mathrm{inter}} L_{\mathrm{inter}}. \end{aligned}$$

With perfect overlap, the optimistic roofline is the slowest resource:

$$T_{\mathrm{ideal}} = \max\left( T_{\mathrm{gpu}}, T_{\mathrm{cpu}}, T_{\mathrm{um}}, T_{\mathrm{nvme}}, T_{\mathrm{intra}}, T_{\mathrm{inter}}, T_{\mathrm{serial}} \right), \qquad R_{\mathrm{target}} \le \frac{1}{T_{\mathrm{ideal}}}.$$

With no overlap, the corresponding latency model is:

$$T_{\mathrm{no\text{-}overlap}} = T_{\mathrm{gpu}} + T_{\mathrm{cpu}} + T_{\mathrm{um}} + T_{\mathrm{nvme}} + T_{\mathrm{intra}} + T_{\mathrm{inter}} + T_{\mathrm{serial}}.$$

Real execution is always slower than the optimistic roof because dependencies, kernel
launches, queueing, cache misses, and resource contention reduce $\eta_r$ and limit
overlap.

#### One DGX Spark

CPU, GPU, CUDA copies, pinned staging, and NVMe DMA all share the same LPDDR system.
They must not be modeled as independent $273\ \mathrm{GB/s}$ domains:

$$\begin{aligned} B_{\mathrm{um}} ={}& B_{\mathrm{dense}} + B_{\mathrm{active\text{-}expert}} + B_{\mathrm{KV}} + B_{\mathrm{head}} \\\\ &+ B_{\mathrm{nvme\text{-}destination}} + B_{\mathrm{staging\leftrightarrow frame}} + B_{\mathrm{temporary}}. \end{aligned}$$

The GPU and CPU ceilings are:

$$\begin{aligned} R_{\mathrm{gpu}} &\le \min\left( \frac{\eta_{\mathrm{gpu}} \cdot 1\ \mathrm{PFLOP/s}}{F_{\mathrm{gpu}}}, \frac{\eta_{\mathrm{um}} \cdot 273\ \mathrm{GB/s}}{B_{\mathrm{gpu}}} \right), \\\\ R_{\mathrm{cpu}} &\le \min\left( \frac{\eta_{\mathrm{cpu}} P_{\mathrm{cpu}}}{F_{\mathrm{cpu}}}, \frac{\eta_{\mathrm{um}} \cdot 273\ \mathrm{GB/s}}{B_{\mathrm{cpu}}} \right). \end{aligned}$$

The advertised FP4 peak is a very loose bound for batch-1 matrix-vector work. NVIDIA
does not publish a directly comparable sustained MoE rate for the 20-core CPU, so
$P_{\mathrm{cpu}}$ must be measured for the selected data type and kernel. CPU and GPU
cannot both claim the full $273\ \mathrm{GB/s}$ simultaneously because they share LPDDR. The current
lossless GB10 path uses the CPU primarily for control and I/O rather than expert math.
A Fiddler-style CPU expert path is useful only when:

$$T_{\mathrm{cpu\text{-}expert}} < T_{\mathrm{stage/copy}} + T_{\mathrm{gpu\text{-}expert}}.$$

The model's all-cold routed-expert payload is known directly from its shape:

$$B_{\mathrm{routed,cold}} = 43 \cdot 6 \cdot 12.75\ \mathrm{MiB} = 3.21\ \mathrm{GiB/target\ pass}.$$

At the observed $10.53\ \mathrm{GiB/s}$ storage ceiling, even perfect all-cold
streaming gives:

$$R_{\mathrm{nvme,cold}} \le \frac{10.53\ \mathrm{GiB/s}}{3.21\ \mathrm{GiB/pass}} \approx 3.28\ \mathrm{target\ passes/s}.$$

That is a storage/model roofline, not an end-to-end Ferrule result. Caching and
prediction reduce $B_{\mathrm{nvme}}$; wrong prefetch increases it.

For a fully staged/resident working set, NVMe can leave the steady critical path, but
LPDDR cannot. Converting the vendor bandwidth to binary units gives:

$$273\ \mathrm{GB/s} \approx 254.25\ \mathrm{GiB/s}.$$

For an illustrative exact pass moving $B_{\mathrm{um}} \in [20,22]\ \mathrm{GiB}$,
the ideal memory-only ceiling is:

$$R_{\mathrm{um}} \le \frac{254.25\ \mathrm{GiB/s}}{B_{\mathrm{um}}} \in [11.6,12.7]\ \mathrm{target\ passes/s}.$$

The exact $B_{\mathrm{um}}$ must come from the S0 target-pass roofline in
[`docs/ROADMAP.md`](docs/ROADMAP.md); 20–22 GiB is an intentionally optimistic
capacity example, not a measured Ferrule token rate. It illustrates why 15–17 full
43-layer passes/s is not the correct single-Spark target.

For DSpark, distinguish checkpoint draft slots $\gamma$, target rows $Q$, accepted
draft tokens, correction/bonus tokens, and mean externally committed output tokens $C$.
The checkpoint declares $\gamma=5$: target verification uses $Q=6$ rows and can commit
at most five accepted drafts plus one correction/bonus token. Widths above six are not
checkpoint-native and require separate proposal/confidence/acceptance evidence. The
release rate is:

$$R_{\mathrm{output}} = \frac{C(Q)}{T_{\mathrm{cycle}}(Q)}.$$

The target-row width is not free. Its resource lower bound must use width-dependent
work and traffic:

$$T_{\mathrm{verify}}(Q) \ge \max\left( \frac{F_{\mathrm{gpu}}(Q)}{\eta_{\mathrm{gpu}}P_{\mathrm{gpu}}}, \frac{B_{\mathrm{um}}(Q)}{\eta_{\mathrm{um}}W_{\mathrm{um}}}, \frac{B_{\mathrm{nvme}}(Q)}{\eta_{\mathrm{nvme}}W_{\mathrm{nvme}}}, T_{\mathrm{verify,serial}}(Q) \right).$$

A complete single-sequence cycle is then:

$$T_{\mathrm{cycle}}(Q) = T_{\mathrm{draft}}(Q) + T_{\mathrm{verify}}(Q) + T_{\mathrm{commit/rollback}}(Q) - T_{\mathrm{safe\ overlap}}(Q).$$

For example, $C=4$ externally committed tokens from a complete
$T_{\mathrm{cycle}}(Q)=250\ \mathrm{ms}$ cycle imply:

$$R_{\mathrm{output}} = \frac{4}{0.25\ \mathrm{s}} = 16\ \mathrm{tokens/s}.$$

This is the architectural meaning of Ferrule's 15–17 tok/s goal; it is a target
operating point, not a current result. It is incorrect in general to multiply the
single-row target-pass roof by accepted draft length, because $F_{\mathrm{gpu}}(Q)$,
$B_{\mathrm{um}}(Q)$, and $B_{\mathrm{nvme}}(Q)$ all depend on target rows and
route overlap.

For routed experts, let $E_{\ell,q}$ be the exact selected-expert set for target row
$q$ at layer $\ell$, and let $S_e=12.75\ \mathrm{MiB}$ be one expert payload. The
all-cold routed bytes for a $Q$-row verification are:

$$B_{\mathrm{routed,cold}}(Q) = S_e \sum_{\ell=1}^{43} \left|\bigcup_{q=1}^{Q} E_{\ell,q}\right|.$$

Because every row selects six distinct experts per layer:

$$3.21\ \mathrm{GiB} \le B_{\mathrm{routed,cold}}(Q) \le 3.21Q\ \mathrm{GiB},$$

until the per-layer union reaches all 256 experts. The lower bound requires every
verified row to select the same experts; the upper bound corresponds to disjoint
selected sets. Under an independent uniform-routing approximation, the expected union
per layer is:

$$\mathbb{E}[U(Q)] = 256\left[1-\left(1-\frac{6}{256}\right)^Q\right].$$

At $Q=4$, this approximation gives about $23.17$ unique experts per layer and:

$$\mathbb{E}\left[B_{\mathrm{routed,cold}}(4)\right] \approx 43 \cdot 23.17 \cdot 12.75\ \mathrm{MiB} \approx 12.40\ \mathrm{GiB/cycle}.$$

If all of those bytes came from the measured $10.53\ \mathrm{GiB/s}$ NVMe path and
$C=4$, the storage-only output-token ceiling would be only:

$$R_{\mathrm{output,nvme}}(4) \le \frac{4 \cdot 10.53}{12.40} \approx 3.40\ \mathrm{tokens/s}.$$

Actual DeepSeek routing is neither uniform nor independent; measured route traces must
supply the real union. The formula shows why wider verification produces useful weight
reuse only when routes overlap or the required experts are already resident.

For a desired output throughput $R^{\star}(\mathrm{output})$, the measured NVMe
path imposes this per-cycle read budget:

$$B^{\star}(\mathrm{NVMe/cycle}) \le \frac{C W(\mathrm{NVMe,observed})}{R^{\star}(\mathrm{output})}.$$

At $C=4$, $R^{\star}(\mathrm{output})=16\ \mathrm{tok/s}$, and
$W(\mathrm{NVMe,observed})=10.53\ \mathrm{GiB/s}$:

$$B^{\star}(\mathrm{NVMe/cycle}) \le \frac{4 \cdot 10.53}{16} \approx 2.63\ \mathrm{GiB/cycle}.$$

Thus 15–17 tok/s is physically plausible only in a cache-heavy speculative regime.
Even the perfect-route-reuse all-cold minimum of $3.21\ \mathrm{GiB/cycle}$ exceeds
the $2.63\ \mathrm{GiB/cycle}$ storage budget for 16 tok/s.

#### One host with multiple GPUs

A DGX Spark contains one GB10; this subsection describes Ferrule's extension to a
separate multi-GPU host rather than multiple internal GPUs in one Spark. For $G$ GPUs
in one host, the ideal node roofline becomes:

$$T_{\mathrm{node}} = \max\left( \frac{F_{\mathrm{gpu}}}{G\eta_{\mathrm{gpu}}P_{\mathrm{gpu}}}, \frac{B_{\mathrm{device}}}{G\eta_{\mathrm{mem}}W_{\mathrm{device}}}, \frac{B_{\mathrm{host}}}{\eta_{\mathrm{host}}W_{\mathrm{host}}}, \frac{B_{\mathrm{nvme}}}{\eta_{\mathrm{storage}}W_{\mathrm{storage,agg}}}, \frac{B_{\mathrm{intra}}}{\eta_{\mathrm{intra}}W_{\mathrm{intra}}} + C_{\mathrm{intra}}L_{\mathrm{intra}}, T_{\mathrm{serial}} \right), \qquad R_{\mathrm{node}} \le \frac{1}{T_{\mathrm{node}}}.$$

Capacity and bandwidth scale differently:

$$M_{\mathrm{resident}} \le \sum_{g=1}^{G} M_g + M_{\mathrm{host,budget}},$$

$$W_{\mathrm{storage,agg}} \le \sum_{d \in \mathrm{independent\ paths}} W_d, \qquad P_{\mathrm{node}} \le \sum_{g=1}^{G} P_g \quad \text{only for partitionable work}.$$

If aggregate GPU memory holds the checkpoint plus KV/arenas, steady expert NVMe traffic
can approach zero. If the model still streams from one drive, adding GPUs does not
remove the storage roof. Tensor parallelism introduces collectives; expert parallelism
reduces weight duplication but introduces dispatch/combine traffic. The scheduler must
choose TP/EP/PP from measured communication and memory costs rather than GPU count.

The ideal linear speedup and parallel efficiency are:

$$S(G) = \frac{R_G}{R_1} \le G, \qquad E(G) = \frac{S(G)}{G} \le 1.$$

#### Multiple nodes with multiple GPUs

For $N$ nodes with $G$ GPUs each:

$$T_{\mathrm{cluster}} = \max\left( \frac{F_{\mathrm{gpu}}}{NG\eta_{\mathrm{gpu}}P_{\mathrm{gpu}}}, \frac{B_{\mathrm{device}}}{NG\eta_{\mathrm{mem}}W_{\mathrm{device}}}, \frac{B_{\mathrm{storage}}}{\eta_{\mathrm{storage}}W_{\mathrm{storage,cluster}}}, \frac{B_{\mathrm{intra}}}{\eta_{\mathrm{intra}}W_{\mathrm{intra}}} + C_{\mathrm{intra}}L_{\mathrm{intra}}, \frac{B_{\mathrm{inter}}}{\eta_{\mathrm{inter}}W_{\mathrm{network}}} + C_{\mathrm{inter}}L_{\mathrm{network}}, T_{\mathrm{serial}} \right), \qquad R_{\mathrm{cluster}} \le \frac{1}{T_{\mathrm{cluster}}}.$$

For expert-parallel decode, a useful first-order activation-traffic estimate is:

$$B_{\mathrm{EP}} \approx 2p_{\mathrm{remote}}Lkbh s,$$

where $L$ is the layer count, $k$ is top-k, $b$ is the batch row count, $h$ is hidden
size, and $s$ is bytes per activation element. The factor two covers dispatch and
returned expert outputs. With $L=43$, $k=6$, $b=1$, $h=4096$, $s=2$ bytes, and
$p_{\mathrm{remote}}=1$:

$$B_{\mathrm{EP}} \approx 2 \cdot 1 \cdot 43 \cdot 6 \cdot 1 \cdot 4096 \cdot 2 = 4{,}227{,}072\ \mathrm{bytes} \approx 4.03\ \mathrm{MiB/target\ pass}.$$

This excludes router metadata and collective overhead. The byte count
is manageable on a fast fabric, but up to 43 latency-sensitive routing rounds can
still dominate small-batch decode. Cross-node tensor parallelism similarly pays
collective latency at many sublayers.

For a Spark cluster, a $200\ \mathrm{Gb/s}$ network path has the vendor-level
unidirectional bandwidth ceiling:

$$W_{\mathrm{network}} \le \frac{200\ \mathrm{Gb/s}}{8} = 25\ \mathrm{GB/s} \approx 23.3\ \mathrm{GiB/s}.$$

Sending the $3.21\ \mathrm{GiB}$ all-cold routed weight set over that path every pass
would cap the weight-transfer roof at:

$$R_{\mathrm{remote\text{-}weight}} \le \frac{23.3\ \mathrm{GiB/s}}{3.21\ \mathrm{GiB/pass}} \approx 7.3\ \mathrm{target\ passes/s}.$$

Expert-parallel placement should therefore keep weights local to their owner and
exchange the MiB-scale activations instead. With independent local NVMe devices and
balanced sharding, the ideal storage ceiling is bounded by:

$$W_{\mathrm{storage,cluster}} \le \sum_{n=1}^{N} W_{\mathrm{nvme},n} \approx N W_{\mathrm{nvme}} \quad \text{only for independent, balanced paths}.$$

One shared storage path does not scale with node count.

Practical scaling order is therefore:

1. use aggregate accelerator memory to eliminate steady weight streaming;
2. prefer expert/pipeline placement that minimizes per-layer cross-node collectives;
3. increase useful batch or verification width to amortize communication latency;
4. replicate storage or shard it across independent paths when streaming remains;
5. report scaling efficiency and externally committed output tokens/s, not only aggregate peak FLOPs.

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

The production reader, bounded selected takeover, compact router handoff, device-global
output-head merge, and GB10 semantic superkernels are implemented. Device-side target-tap
capture and fused projection/normalization execute against the real checkpoint. Independent
committed DSpark context states feed hybrid attention over paged history plus the ephemeral
proposal block, while proposal heads, sequential Markov selection, and confidence execute
inside the prepared CUDA/CUTLASS image. The production server now carries this proposal
through exact target verification, correction/bonus handling, rollback/replay, and external
output reconciliation.

The next optimization order is:

1. export official Python fixtures and localize proposal numerical differences;
2. prove proposal/logit/confidence and state-transition parity;
3. reconcile scheduler predictions and admission with actual physical expert operations;
4. execute all-hit routed layers without host materialization and remove hot-path allocation;
5. capture stable graphs only after correctness and I/O accounting are exact;
6. rerun the production serving and release suite.

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
| **Executable model plan** | Prepare-time binding selects immutable buffers, launch descriptors, and the kernel provider; the token hot path performs no architecture discovery. |
| **CUTLASS provider boundary** | GB10 semantic superkernels use pinned CUTLASS/CuTe primitives behind a versioned POD ABI; Rust retains stream, memory, plan, KV, and residency ownership. |
| **Native packed execution** | `ExecutionBatch` represents continuous-batch `B × Q=1` decode and one-sequence `Q=V` DSpark verification without a CPU execution plan. |
| **Expert-I/O scheduling** | Admission accounts for incremental expert bytes, inflight reads, pinned slabs, uploads, deadlines, and current residency instead of batching by row count alone. |
| **Transactional KV and DSpark** | Physical multi-plane KV supports branch, rollback, and accepted-prefix replay for exact speculative verification. |
| **Single-owner safetensors ingest** | DSV4 tensors and expert extents are bound with bounded reads rather than loading the checkpoint into RAM. |
| **Narrow validation anchors** | CPU/reference code is retained only where it acts as a numerical oracle for CUDA correctness. |

---

## System vision

Ferrule is intentionally a narrow DSV4/CUDA appliance. The runtime owns scheduling,
KV transactions, expert residency, and exact speculative state; the prepared model owns
DSV4 math and immutable execution buffers; the selected kernel provider owns launches.

```text
request-centric scheduler
  -> executable DSV4 plan
  -> GB10 CUTLASS/CuTe semantic superkernels + dedicated sparse/control kernels
  -> transactional KV and expert residency
```

GB10 / `sm_121a` is the only supported production target. Other hardware fails explicitly;
a future target will receive an independent performance path.

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

`just build-cuda` automatically fetches and verifies the pinned CUTLASS 4.6.1 checkout on
the first build.

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
| Real large-model milestone | The exact CUDA target, checkpoint-native DSpark proposal, packed verification, correction/bonus transaction, and production OpenAI serving path are connected. Official proposal parity, acceptance, full-cycle residency/I/O coordination, and release throughput remain open. |
| Execution ABI | E1–E4 complete: public `ExecutionBatch`, prepared-executor traits, runtime-private `ScheduledBatch`, strict validation, and native packed multi-session execution |
| DSV4 execution boundary | The model owns HC/MLA/compressor/router/MoE math; runtime remains model-neutral and owns scheduling plus logical page lifecycle through generic traits |
| Expert streaming | Immutable source catalogs plus production `O_DIRECT + io_uring` fixed-file reads into registered CUDA-pinned slabs; CUDA promotes selected/on-time experts into ordinary device frames governed by stable slots, generations, leases, bounded admission, and selected takeover. |
| Expert residency | Runtime-owned stable slots, generations, leases, host/pinned budgets, upload admission, and dense residency snapshots feed the DSV4 expert-I/O advisor |
| Attention | OLMoE GQA executable; DSV4 MLA sparse attention correctness-first CUDA path |
| Hyper-connections | DSV4 HC source binding + reference `hc_pre`/`hc_post`/`hc_head` |
| KV cache | `KvPageManager` plus bounded CUDA multi-plane pools, paged attention/indexer kernels, branch, rollback, COW, preempt/restore, and DSpark accepted-prefix replay |
| Quantization | Q4_0/Q8_0, FP4 E2M1 + E8M0 scales, FP8 E4M3FN, mixed precision policy |
| WeightPack | mmap'd reader, streaming writer, zero-copy slices, WeightPack-only load path |
| Kernel plan | One semantic `LayerKernelPlan` per layer binds the GB10 provider; M scheduling remains provider-private and missing capabilities fail explicitly |
| Generation | Device/global top-k candidate selection, deterministic greedy commit, stop strings, EOS handling, and incremental token text; unsupported sampling is rejected at the API boundary |
| Chat templates | OLMoE, ChatML, Llama3, Qwen, DeepSeek-V4, Plain |
| Serving | `ferrule-server` + `ferrule serve`: Axum/Hyper/Tokio, dedicated model-owner thread, bounded queues, disconnect cancellation, `/health`, `/v1/models`, `/v1/chat/completions`, and `/v1/completions`; reusable `vllm bench serve` concurrency workflows; greedy only for now. |
| Speculation | The checkpoint-native DSpark block contract is frozen. Device-side target context, non-causal proposal execution, proposal heads, exact packed verification, draft-prefix acceptance, correction/bonus handling, rollback/replay, and external-output reconciliation are connected. Official fixture parity and measured acceptance remain open. |
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
cargo install --locked cargo-nextest # One-time local test runner setup
just test           # CPU tests via nextest, doctests, optional CUDA tests
just test-runtime   # Targeted runtime tests with cargo test
just test-cuda      # CUDA tests via cargo oxide when available
just fmt            # Format check
just clippy         # CPU workspace/all-target Clippy with warnings denied
just clippy-cuda    # CUDA feature lint on a configured CUDA host
just lint           # fmt + CPU clippy + strict Rustdoc
just miri           # Runtime library tests under Miri
just deny           # Advisories, licenses, bans, and source policy
just coverage       # CPU HTML/LCOV reports via llvm-cov + nextest
```

CI publishes the Nextest JUnit report as a GitHub Check and downloadable
artifact. CPU coverage is enforced at a 60% line baseline, reported by Codecov
on pull requests, and uploaded as browsable HTML plus an LCOV artifact.

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
| [Scheduler Architecture](docs/scheduler-architecture.md) | Segment-scoped exact routing, global expert loads, UMA credits, draining, and deterministic simulation |
| [CUTLASS](docs/CUTLASS.md) | GB10 semantic provider ABI, pinned dependency setup, validation, and benchmark contract |
| [Serving](docs/serving.md) | Axum/Hyper/Tokio model-worker ownership, OpenAI/SSE contract, cancellation, admission, and benchmark workflows |
| [Roadmap](docs/ROADMAP.md) | Current F1/F2/F3 status, remaining MTP/I/O/graph critical path, and release contract |
| [Expert Memory & Telemetry](docs/expert-memory-architecture.md) | Host/pinned expert budgets, stable device residency, GB10 constraints, and benchmark-safe telemetry |

---

## License

Apache-2.0
