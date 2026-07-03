# Ferrule Storage and Residency Architecture

## Status

Draft design. This document defines the storage/residency abstraction that should replace scattered concepts such as artifact tensor reads, expert load sources, host staging, GPU residency, WeightPack chunks, future io_uring prefetch, and remote cache systems.

The immediate goal is **not** to build a complex storage engine. The goal is to give all storage-related code one vocabulary so the first high-ROI optimizations can stay simple:

1. keep useful expert weights resident,
2. stage likely-needed expert weights in host memory before decode needs them,
3. later overlap disk/network read, host staging, H2D upload, and CUDA compute.

## Design pressure

Current DSV4 counters show the bottleneck clearly:

- many selected experts are loaded synchronously during decode,
- expert payloads move local disk → host bytes → GPU handles on the hot path,
- eviction policy can thrash resident experts,
- artifact tensors, expert slices, WeightPack chunks, KV state, and CUDA buffers use different terms and ownership boundaries.

The storage abstraction must support DSV4, but it must not become DSV4-specific.

## Core idea

Ferrule should treat every loadable or resident runtime datum as a **storage object** with:

1. **identity** — what logical object is this?
2. **layout** — how are bytes interpreted?
3. **locators** — where can the bytes be obtained?
4. **replicas** — where does the object currently reside?
5. **policy** — how important is it to keep/prefetch/evict?
6. **transfers** — what movement is needed to satisfy execution?

The system should only distinguish two storage domains:

```text
Local domain:
  device memory  (GPU VRAM / accelerator memory)
  host memory    (DRAM, optionally pinned)
  local disk     (filesystem, mmap, NVMe, WeightPack, safetensors)

Remote domain:
  anything outside the local machine boundary
  object store, LAN cache, RDMA service, Mooncake-like cache, HiCache-like cache, HTTP/S3/etc.
```

Everything else is a backend detail.

## Vocabulary

### Storage object

A logical object that can be located, cached, moved, and referenced by execution.

Examples:

- an artifact tensor slice,
- one row range of `lm_head`,
- one routed expert matrix,
- one routed expert bundle,
- one WeightPack chunk,
- one KV page,
- one decode arena buffer,
- one graph external object.

Sketch:

```rust
pub struct StorageObjectId(String);

pub enum StorageObjectKind {
    ArtifactTensor,
    ArtifactTensorRows,
    ExpertMatrix,
    ExpertBundle,
    WeightPackChunk,
    KvPage,
    DecodeArenaBuffer,
    GraphExternal,
    Opaque,
}

pub struct StorageObjectDescriptor {
    pub id: StorageObjectId,
    pub kind: StorageObjectKind,
    pub bytes: u64,
    pub layout: StorageLayout,
    pub mutability: StorageMutability,
}
```

### Layout

Layout is semantic enough for validation and backend selection, but not model-family-specific.

```rust
pub enum StorageLayout {
    Bytes,
    Tensor(TensorLayout),
    ExpertBundle(ExpertBundleLayout),
    KvPage(KvPageLayout),
    Opaque { tag: String },
}
```

DSV4-specific meaning stays outside this layer. For example, “this object is DSV4 layer 17 expert 42” belongs to model-family binding metadata; storage only sees an expert bundle object with size/layout/locators.

### Placement

Placement describes where a replica is, not where it originally came from.

```rust
pub enum Placement {
    Local(LocalPlacement),
    Remote(RemotePlacement),
}

pub enum LocalPlacement {
    Device { device_id: u32, memory: DeviceMemoryKind },
    Host { pinned: bool },
    Disk { volume: Option<String> },
}

pub enum DeviceMemoryKind {
    Vram,
    Unified,
    Other(String),
}

pub struct RemotePlacement {
    pub endpoint: RemoteEndpoint,
    pub region: Option<String>,
}
```

Important rule:

- `Gpu`, `Cpu`, `HostMmap`, `LocalShard`, `WeightPackChunk`, `Remote` should converge into this `Placement` vocabulary.
- Avoid using `Source` for runtime residency. Use **locator** for where bytes may be fetched and **placement** for where bytes currently reside.

### Locator

A locator says how an object can be fetched if no suitable replica is resident.

```rust
pub enum ObjectLocator {
    LocalFile { path: PathBuf, offset: u64, bytes: u64 },
    LocalMmap { path: PathBuf, offset: u64, bytes: u64 },
    WeightPack { path: PathBuf, chunk: String, offset: u64, bytes: u64 },
    RemoteObject { uri: String, offset: u64, bytes: u64 },
    RemoteCache { key: String, bytes: u64 },
}
```

Locators are not execution handles. They are catalog entries.

### Replica

A replica is a current resident copy of an object.

```rust
pub struct ObjectReplica {
    pub object: StorageObjectId,
    pub placement: Placement,
    pub bytes: u64,
    pub state: ReplicaState,
    pub generation: u64,
}

pub enum ReplicaState {
    Ready,
    Loading,
    Evicting,
    Failed { reason: String },
}
```

An object may have several replicas simultaneously:

```text
expert bundle E:
  local disk locator: safetensors shard extents
  host replica: staged bytes
  device replica: CUDA expert handles
```

## Storage graph

Ferrule should model movement as transitions through this graph:

```text
Remote
  ↓ upload/fetch/cache
Local Disk
  ↓ read/mmap/io_uring
Host Memory
  ↓ H2D upload / async copy
Device Memory
```

The graph is conceptual. Backends may skip edges when supported:

- local file → pinned host buffer,
- local file → GPU direct storage later,
- remote cache → local disk,
- remote cache → host memory,
- host memory → device memory on a CUDA stream.

## Why prefill/decode makes prediction possible

Large-model inference has two different execution regimes, and they expose different
storage opportunities.

### Prefill

During prefill, the full prompt token sequence is known. The model still executes
layers sequentially, but each layer often processes many prompt tokens at once.
This creates a wider prediction window than single-token decode:

- token positions are known,
- sequence length and attention/KV growth are known,
- dense layer order is fixed,
- once a layer's router logits are computed, all selected experts for that layer's
  prompt tokens are known,
- expert popularity for the prompt segment can be counted before launching that
  layer's expert kernels.

Prefill can therefore prewarm or batch storage at layer granularity:

```text
prefill layer L
  compute router logits for all prompt tokens
  determine selected expert set / histogram
  ensure hot selected experts are device-resident
  stage next likely experts in host memory if budget allows
  execute MoE for the prompt segment
```

For dense tensors, the execution order is deterministic, so prefetch is not
prediction in the statistical sense: it is simply the known layer schedule.

### Decode

During decode, only one new token is produced at a time. Exact future activations
are not known until the previous token has been sampled and each layer has run far
enough to compute router logits. This narrows the prediction window, but it does
not eliminate it:

- the next token id is known immediately after sampling,
- early hash/router-table layers may have exact expert ids from the token id,
- score-top-k layers know exact experts only after their router logits are computed,
- recent expert usage is often sticky enough for LRU/hot-retain policies to help,
- router logits can provide near-miss candidates: selected top-k plus next-best
  candidates are useful prefetch hints,
- layer order is fixed, so dense weights and per-layer scratch/KV placement can be
  scheduled ahead of compute.

Decode prediction is therefore a mix of exact, cheap, and statistical signals:

| Signal | Exact? | Useful for |
|---|---:|---|
| sampled next token id | yes | hash-router expert prewarm, token-dependent metadata |
| layer execution order | yes | dense tensor / scratch / KV scheduling |
| current layer router selected top-k | yes, once logits exist | execute-now expert residency |
| router next-best candidates | heuristic | host-stage / device-prefetch candidates |
| recent expert history | heuristic | retain hot device experts |
| prompt/domain/session history | heuristic | cross-token expert cache priority |
| prefix/KV reuse metadata | yes/heuristic | KV page residency and remote KV cache lookup |

The key design point: **prediction does not have to be perfect**. A missed
prediction falls back to synchronous load. A correct prediction removes disk read,
remote fetch, or H2D upload from the critical path.

## What should be prewarmed or demoted?

Not all tensors are equally worth tiering. Ferrule should treat each object kind
differently while using the same storage vocabulary.

| Object kind | Prewarm candidate? | Demotion candidate? | Notes |
|---|---:|---:|---|
| Routed expert bundles | high | high | Best first target for DSV4 local throughput. Only top-k experts are used per token, but the model has many experts. |
| Shared experts | medium | low/medium | Used every layer/token when present; usually better kept device-resident if memory allows. |
| Dense attention/MLP linears | medium | medium | Used every token in fixed layer order. If model exceeds VRAM, layer-wise streaming can prefetch them deterministically. |
| Output head chunks | medium | low/medium | Full-vocab/top-k scans touch chunks in fixed order; cache chunks and avoid repeated shard reads/uploads. |
| KV pages / compressed KV / indexer KV | high for reuse | medium/high under pressure | Recent context should stay device-resident; older pages can spill to host/disk/remote at latency cost. |
| Decode arena activations | low | no | Ephemeral. Usually recompute or discard; demoting them is rarely useful. |
| Prompt/prefix KV cache | high | high | Good candidate for hierarchical cache: device → host → local disk → remote. |
| Adapters / LoRA / optional tensors | workload-dependent | high | Only keep active adapters or hot variants resident. |

### Expert bundles are the first ROI target

For a MoE model such as DSV4, an expert bundle is large enough that synchronous
read/upload hurts, but sparse enough that not every expert is needed every token.
That makes experts ideal for a tiered residency policy:

```text
Device resident:
  selected experts needed now
  recently hot experts while budget allows

Host staged:
  predicted experts for next token/layer
  next-best router candidates
  remote/local bytes already fetched but not uploaded

Local disk / remote:
  cold experts
  fallback locators
```

A simple v1 policy can be effective:

1. selected experts must be device-resident,
2. predicted experts should be host-staged,
3. hot resident experts should be retained while device slots allow,
4. coldest resident experts are demoted first,
5. if prediction misses, synchronously load and count the miss.

### Demotion means removing a replica, not changing the object

“Demote to memory/disk” should mean:

- keep the logical `StorageObjectId`,
- drop or downgrade one `ObjectReplica`,
- preserve another locator or lower-tier replica,
- update residency counters.

For immutable weights, demotion is cheap semantically: the object can always be
reloaded from a locator. For mutable KV state, demotion must preserve the latest
generation and requires stricter correctness checks.

## Have other inference systems already solved this?

The short answer: **parts of this are production-standard, but dynamic expert
prediction plus hierarchical local/remote storage is still an active systems area,
especially for large MoE models on constrained local hardware.**

### What is common today

- **KV cache residency and paging** are common. vLLM popularized paged KV memory
  management; SGLang adds prefix/radix-style reuse; newer systems add hierarchical
  KV caches that can spill or share KV across requests.
- **Prefix cache / prompt cache** is common in high-throughput serving. Engines try
  to avoid recomputing or reloading reusable prefix KV.
- **Static or layer-wise weight offload** exists. llama.cpp supports CPU/GPU layer
  placement and mmap-style loading. FlexGen / DeepSpeed-style offload systems move
  weights/KV/activations across GPU, CPU, and NVMe for models larger than VRAM.
- **MoE expert parallelism and batching** are common in server systems. vLLM,
  TensorRT-LLM, SGLang, and vendor stacks focus on placing experts across GPUs,
  batching selected experts, and balancing expert load.

### What is less uniformly solved

- **Per-token expert prediction from local disk/host staging** is not a universal
  mainstream serving feature. Large server deployments usually prefer enough GPU
  memory or expert parallelism so experts are already distributed/resident.
- **Single-machine MoE with model larger than VRAM** often still needs custom
  offload, hot-expert caching, or research-system techniques.
- **Remote hierarchical caches** such as Mooncake-like or HiCache-like designs are
  closer to active infrastructure work than a solved local-runtime primitive. They
  are valuable, but they should plug in as storage backends, not shape model
  semantics.

### Useful lessons from existing systems

- vLLM lesson: make memory blocks explicit and schedulable; do not hide KV memory
  behind ad-hoc vectors.
- SGLang lesson: prefix/reuse metadata matters as much as raw kernel speed.
- llama.cpp lesson: mmap/local file layout and partial offload are practical and
  simple, but not enough for dynamic MoE expert staging.
- FlexGen/DeepSpeed offload lesson: a planned GPU/CPU/NVMe schedule can run models
  bigger than VRAM, but overlap and batching determine whether it is usable.
- PowerInfer-like lesson: activation sparsity/hotness can guide which parts of a
  model deserve GPU residency.
- TensorRT-LLM/vendor lesson: if all hot weights fit in GPU memory and requests are
  batched, expert parallelism beats complicated local disk staging; Ferrule needs
  staging because the target local DSV4 path is memory-constrained.

Ferrule should therefore implement the general primitive, not copy one system:

```text
predict or know future object demand
  → request residency at a target placement
  → satisfy from best available replica/locator
  → overlap transfer where possible
  → evict/demote cold replicas under budget
```

## Ferrule-specific prediction strategy

Ferrule should start with low-complexity, measurable predictors.

### Exact predictors

- Dense layer schedule: layer order is fixed for prefill/decode.
- Output-head chunk order: chunk scans are fixed for greedy/top-k decode.
- KV page growth: token position and context window policy are known.
- Hash-router expert ids: when the router uses token-id tables, selected ids can be
  known as soon as the sampled token id is known.

### Cheap heuristic predictors

- Router near-miss experts: selected top-k plus next `m` candidates from router
  scores.
- Recent expert hotness: retain last-used experts while device budget allows.
- Prompt prefill histogram: for a prompt segment, pre-stage experts that appear
  frequently in the layer's router decisions.
- Session-local hot set: keep experts that repeatedly activate in the same session.

### What Ferrule should not predict first

- full future token sequences,
- deep future layer activations before computing preceding layers,
- remote-cache hit probabilities without counters,
- expensive learned predictors before simple LRU/top-k+margin signals are measured.

## Component boundaries

### 1. Storage catalog

Owns object descriptors and locators.

Responsibilities:

- map semantic artifacts to storage objects,
- register local and remote locators,
- normalize safetensors / WeightPack / cache chunk metadata,
- answer “where can object X be fetched from?”

Non-responsibilities:

- no scheduling,
- no prefetch decisions,
- no CUDA handles.

Sketch:

```rust
pub trait StorageCatalog {
    fn descriptor(&self, id: &StorageObjectId) -> Option<&StorageObjectDescriptor>;
    fn locators(&self, id: &StorageObjectId) -> &[ObjectLocator];
}
```

### 2. Residency manager

Owns current replicas and budgets.

Responsibilities:

- answer “is object X available at placement Y?”,
- track host/device/disk/remote replicas,
- enforce budgets,
- choose evictions when policy asks for space,
- expose telemetry.

Sketch:

```rust
pub struct ResidencyRequest {
    pub object: StorageObjectId,
    pub desired: PlacementClass,
    pub priority: ResidencyPriority,
    pub deadline: Option<Instant>,
    pub reason: ResidencyReason,
}

pub enum PlacementClass {
    Device,
    Host,
    LocalDisk,
    Remote,
}

pub enum ResidencyReason {
    ExecuteNow,
    Prefetch,
    RetainHot,
    Debug,
}
```

### 3. Transfer engine

Executes movement plans.

Responsibilities:

- blocking disk read,
- mmap read,
- future io_uring read,
- remote fetch,
- host staging,
- H2D/D2H copies,
- async CUDA stream copy later,
- update counters.

Sketch:

```rust
pub trait TransferEngine {
    fn ensure(&mut self, request: ResidencyRequest) -> Result<ResidencyHandle>;
    fn prefetch(&mut self, requests: &[ResidencyRequest]) -> Result<Vec<TransferTicket>>;
    fn poll(&mut self) -> Result<Vec<TransferEvent>>;
}
```

First implementation can be synchronous and boring. The trait exists so io_uring / pinned memory / remote cache can be added later without changing planner semantics.

### 4. Policy / planner

Decides what should be resident or staged.

Responsibilities:

- rank current selected objects,
- rank predicted objects,
- retain hot objects,
- choose prefetch candidates,
- choose eviction candidates,
- stay generic over object kind.

Sketch:

```rust
pub struct ResidencyPolicy {
    pub device_budget_bytes: u64,
    pub host_staging_budget_bytes: u64,
    pub local_disk_cache_budget_bytes: Option<u64>,
    pub retain_hot: bool,
    pub prefetch_window: usize,
}

pub struct ResidencyScore {
    pub execute_now: bool,
    pub predicted: bool,
    pub last_used_step: u64,
    pub load_cost_bytes: u64,
    pub object_bytes: u64,
}
```

The first policy should be simple:

1. selected objects must be device-resident,
2. predicted objects should be host-staged or device-resident if space remains,
3. hot resident objects should be kept while budget allows,
4. evict coldest objects first.

No async priority queue is required for v1.

## How current concepts map

| Current concept | New concept |
|---|---|
| `ArtifactTensorSlice` | `StorageObjectDescriptor + ObjectLocator::LocalFile` |
| `ArtifactTensorReader` | local disk transfer backend |
| `ExpertLoadSource` | object locators / legacy compatibility wrapper |
| `ExpertStorageTier` | `Placement` / `PlacementClass` |
| `ResidentExpertHandle` | device replica handle for an expert bundle |
| `CpuExpertHandleStore` | host/device replica registry, later split by placement |
| `WeightPackChunk` | `ObjectLocator::WeightPack` |
| CUDA linear cache | device replica cache for artifact linear object |
| KV cache page | mutable local device/host object |
| remote expert/cache URI | `ObjectLocator::RemoteObject` or `RemoteCache` |
| Mooncake / HiCache | remote/local cache backend implementing locator + transfer |

## Expert-specific flow after refactor

### Current hot path

```text
router selects experts
  → planner asks for selected experts
  → sync read local safetensors slices
  → build ExpertComputeBundle
  → upload to CUDA handles
  → execute
  → maybe evict immediately
```

### Target v1 flow

```text
router selects experts
  → planner emits selected + predicted storage requests
  → residency manager checks device replicas
  → selected miss:
       host-staged hit → upload to device
       host-staged miss → blocking read → host stage → upload
  → predicted objects are staged if budget allows
  → hot resident experts are retained while slots/budget allow
```

This gives immediate ROI without async complexity.

### Target v2 flow

```text
previous step prediction
  → background prefetch reads predicted experts into host pinned memory
current step selection
  → selected expert usually host-staged or device-resident
  → H2D upload can overlap on a transfer stream
  → compute stream waits only when needed
```

### Target v3 flow

```text
remote/disk cache hierarchy
  → local disk cache / Mooncake-like cache / HiCache-like cache
  → host staging
  → async H2D
  → device-resident execution
```

The planner does not need to know whether bytes came from safetensors, WeightPack, remote object store, or a LAN cache.

## Relationship with runtime graph

Runtime graph should not contain concrete storage backend details.

Graph-visible externals should reference storage object IDs or semantic backend object IDs:

```text
ExternalBindingKind::ArtifactTensor  → StorageObjectId
ExternalBindingKind::ExpertRegistry  → collection of StorageObjectId
ExternalBindingKind::KvState         → mutable storage object / state handle
ExternalBindingKind::ResidentExpert  → device placement requirement, not concrete CUDA handle
```

Backend lowering turns object IDs and placement requirements into actual handles.

## Relationship with CUDA

`ferrule-cuda` should expose device-handle primitives and transfer counters, but it should not own model-family storage policy.

CUDA responsibilities:

- upload host bytes to device buffers / handles,
- later support pinned buffers and async copy streams,
- expose device replicas as opaque handles,
- report H2D/D2H bytes, copy counts, kernel launches.

Runtime storage responsibilities:

- decide what to stage / retain / evict,
- choose local vs remote locator,
- connect object IDs to CUDA uploads.

## Minimal implementation plan

### Phase 0 — vocabulary only

Add storage/residency types without changing execution behavior:

- `StorageObjectId`
- `StorageObjectDescriptor`
- `ObjectLocator`
- `Placement`
- `ObjectReplica`
- `ResidencyRequest`
- `ResidencyPriority`

Keep compatibility adapters for existing `ArtifactTensorSlice` and `ExpertLoadSource`.

### Phase 1 — expert residency simplification

Use the new policy vocabulary to express what is already needed:

- selected experts: `ExecuteNow → Device`
- predicted experts: `Prefetch → Host` or `Device if budget remains`
- hot resident experts: `RetainHot → Device`

Keep implementation synchronous.

### Phase 2 — host-staged expert cache

Add a small host memory cache keyed by `StorageObjectId`:

```text
local disk locator → host replica bytes
host replica bytes → device expert handle
```

Acceptance:

- selected expert read path can hit host cache,
- JSON counters distinguish disk reads, H2D uploads, device resident hits,
- no async runtime required.

### Phase 3 — prediction inputs

Feed cheap prediction hints into policy:

- current router selected top-k,
- optional next-best router candidates for score-top-k layers,
- hash-router layers can start with retain-only behavior.

Acceptance:

- predicted host-stage hit rate is visible,
- misses fall back to blocking read.

### Phase 4 — asynchronous backends

Add transfer backends behind the same trait:

- threaded local reader,
- io_uring local reader on Linux,
- pinned host staging,
- async H2D transfer stream.

Acceptance:

- no runtime/model API changes when switching transfer backend.

### Phase 5 — remote/cache systems

Add remote/cache locators and transfer backends:

- local disk cache,
- remote object store,
- Mooncake-like distributed cache,
- HiCache-like prefix/expert cache.

Acceptance:

- remote systems are just locator/transfer implementations,
- graph/model/operator code does not branch on remote backend names.

## What not to do now

- Do not add `io_uring` as the first abstraction.
- Do not add DSV4-specific storage APIs.
- Do not put CUDA handles into graph IR nodes.
- Do not make remote cache semantics leak into expert routing.
- Do not rename everything in one patch; introduce compatibility adapters first.
- Do not build a complex priority queue until counters show host staging alone is insufficient.

## Near-term recommendation

The next patch should implement only:

1. common storage vocabulary types,
2. adapters from existing artifact/expert descriptors,
3. a tiny host-staged expert cache interface,
4. counters for disk-read bytes/cache hits/H2D uploads,
5. no async runtime yet.

That gives a clean path to io_uring and Mooncake/HiCache-style systems later while keeping the first optimization small and measurable.
