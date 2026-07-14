# Expert Memory and Telemetry Architecture

_Status: owner-thread host/pinned cache budgets implemented; alternative GB10 backing modes require A/B validation._

_Last updated: 2026-07-13_

## 1. Scope

This document defines the model-neutral memory boundary used by streamed MoE experts.
It complements [`storage-residency-architecture.md`](storage-residency-architecture.md):
storage policy decides what should be resident, while this document defines how local
retention pools account for and enforce memory.

The current DSV4 physical flow is:

```text
immutable expert source catalog
  -> mmap/safetensors whole-expert read
  -> pageable HostStagedExpertCache
  -> pinned CudaPinnedExpertCache or transient pin
  -> asynchronous CUDA upload ticket
  -> runtime-selected stable device slot
  -> six device buffers + stable pointer/generation tables
```

KV paging is independent. Expert weights are retained as whole compute bundles; they
are not physical weight pages.

## 2. Neutral vocabulary

`ferrule-common::memory` owns:

- `MemoryTopology`: `Discrete` or `CoherentUnified` hardware;
- `MemoryPoolKind`: stable classification of resident, staged, pinned, transient, and
  file-page-cache pools;
- `MemoryPoolLimits`: simultaneous entry and retained-payload byte limits;
- `MemoryPoolStats`: limits, current/peak bytes, entries, hits/misses, admissions,
  evictions, and rejections;
- `OwnerMemoryLru<K, V>`: synchronization-free O(1)-average owner-thread admission,
  lookup, promotion, accounting, and per-entry eviction.

These types contain no DSV4 tensor names, shapes, or CUDA handles. Other model
adapters can compose the same policy and cache primitive.

`ferrule-model::moe::ExpertMemoryPolicy` combines two neutral limits:

- pageable whole-expert host retention;
- pinned whole-expert host retention.

The CUDA pinned payload remains backend-private. Only its limits and statistics cross
the model/backend boundary.

## 3. Ownership and hot-path rules

Both caches are mutated only by the dedicated model-owner thread. Background workers
may read artifact slices and return completed bundles through the existing channel,
but they do not publish cache state.

The hot path has:

- no `Mutex`, process-global allocator, or per-hit atomics;
- no linear `VecDeque::retain` promotion;
- no full-cache byte scan;
- no deep clone of `ExpertComputeBundle` on a host-cache hit;
- oversized-entry rejection before an existing same-key resident is removed;
- simultaneous entry/byte enforcement with explicit rejection counters;
- a one-shot async-completion handoff, so rejected long-term retention still feeds the
  pending upload instead of causing duplicate or repeated artifact reads.

`Arc<ExpertComputeBundle>` shares pageable expert payloads between the cache and the
current upload operation. Eviction removes cache retention; an in-flight reference may
keep the allocation alive until the operation completes, which is why pool bytes are
retained-payload accounting rather than process RSS.

## 4. Serving configuration

`ferrule serve` freezes these options before the model worker starts:

```text
--expert-host-cache-entries <COUNT>    default 64; 0 disables retention
--expert-host-cache-mb <MIB>           default 1024; 0 means entry-limited only
--expert-pinned-cache-entries <COUNT>  default 16; 0 disables retention
--expert-pinned-cache-mb <MIB>         default 256; 0 means entry-limited only
```

MiB conversion is checked; overflow is rejected. The resulting
`ExpertMemoryPolicy` is stored in `DeepSeekV4PrepareOptions` and passed through the
prepared runner/operator construction path to the actual host and pinned caches.
There is no second DSV4 environment-variable capacity source.

Example bounded GB10 starting point (a tuning input, not a universal optimum):

```bash
./target/release/ferrule serve models/DeepSeek-V4-Flash-DSpark \
  --expert-host-cache-entries 64 \
  --expert-host-cache-mb 1024 \
  --expert-pinned-cache-entries 16 \
  --expert-pinned-cache-mb 256
```

The correct values depend on model expert size, active KV capacity, fixed-weight
residency, OS page cache, and concurrency. KV has an independent enforced serving
budget (`--kv-cache-mb`, default 1024 MiB); expert-cache budgets do not borrow from it.
Benchmark sweeps must report the selected limits and current/peak/rejection statistics.

## 5. GB10 / coherent unified memory

DGX Spark exposes 128 GiB coherent unified LPDDR5x to CPU and GPU. Pageable host,
pinned host, device allocations, file page cache, KV pools, arenas, and upload
retirement therefore compete for one physical capacity and memory bandwidth even when
CUDA presents different allocation APIs.

For the current DSV4 shape, one whole routed expert is approximately 12.75 MiB. An
entry-only 256/64 host/pinned policy can retain roughly 3.2 GiB plus 0.8 GiB before
allocator overhead and in-flight references. Byte limits are therefore the authoritative
capacity safety control on GB10; entry limits remain useful for metadata and LRU bounds.

Ferrule does **not** yet assume that mapped pinned direct reads or managed expert
backing outperform explicit asynchronous uploads. The required A/B matrix is:

1. current device-buffer residency;
2. mapped pinned direct-read backing;
3. managed or preallocated stable-slot backing.

Each variant must hold model, prompts, residency policy, cache limits, and quality
constant, and report TTFT, ITL, throughput, memory bandwidth, page faults, transfer
waits, and retained/peak pool bytes. A different backing mode is admitted only if it
wins measured end-to-end workloads without weakening stable-slot and failure-atomic
publication semantics.

## 6. Benchmark-safe telemetry

Headline benchmarks run with DSV4 profiling disabled by default:

```text
FERRULE_DSV4_PROFILE=false
FERRULE_DSV4_PROFILE_SYNC=false
FERRULE_CUDA_MOE_TIMING=false
```

With profiling off, DSV4 layer/attention hot paths do not sample `Instant`, do not
perform profiling synchronizations, and do not populate timing maps. CUDA dispatch
environment controls are parsed once when the CUDA operator context is created.
Memory/cache and correctness counters remain owner-thread integer updates and are read
as snapshots; no log formatting occurs in the model loop.

Diagnostic synchronized profiling is intentionally expensive and must never be used
for headline throughput or latency. Future production metrics should publish periodic
owner-thread snapshots to the HTTP side. Sampled CUDA timing should use a preallocated
event pool and a bounded non-blocking ring; a full ring drops samples rather than
blocking execution. Per-token/per-layer formatted logs and synchronous JSON formatting
on the model thread are prohibited.

## 7. Remaining memory work

- add an enforced transient-upload headroom budget rather than exposing an ignored
  configuration value;
- add global resident-expert byte admission once runtime can enforce it alongside
  per-layer stable-slot capacity;
- account for duplicate physical backing and allocator overhead in platform-specific
  process metrics;
- A/B mapped/managed backing on GB10 before changing the default transfer path;
- expose periodic production metrics without adding synchronization to decode.
