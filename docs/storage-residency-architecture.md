# Ferrule Storage and Residency Architecture

## Status

Rewrite of the storage/residency design. The earlier version was structured as a
paper survey with FASTER/F2/Mooncake/HiCache/LMCache parallels inline; this
revision puts Ferrule's code reality and implementation plan first and collects
external-system mappings into a single appendix.

The core abstraction is unchanged and still stands:

```
StorageObjectId   — logical object identity
ObjectLocator     — where bytes can be fetched from
Placement         — where a replica currently resides
ObjectReplica     — a resident copy at a placement
ResidencyManager  — current replicas, budgets, hit/miss
TransferEngine    — disk read, mmap, H2D, future async/RDMA
Policy            — decide retain / prefetch / evict
```

What changed is the framing: this is now a Ferrule implementation plan, not a
paper-informed architecture essay.

---

## Table of Contents

1. [Problem statement](#problem-statement)
2. [Goals and non-goals](#goals-and-non-goals)
3. [Current code reality](#current-code-reality)
4. [Unifying the two systems](#unifying-the-two-systems)
5. [Target vocabulary](#target-vocabulary)
6. [Component boundaries](#component-boundaries)
7. [V1 scope](#v1-scope)
8. [Expert residency flow](#expert-residency-flow)
9. [Integration points](#integration-points)
10. [Metrics and validation](#metrics-and-validation)
11. [Implementation plan](#implementation-plan)
12. [Risks and invariants](#risks-and-invariants)
13. [External systems mapping](#external-systems-mapping)
14. [References](#references)

---

## Problem statement

Ferrule targets sparse MoE inference under memory pressure. The immediate
pressure test is DeepSeek V4 (DSV4), where routed experts dominate memory and
cannot all stay device-resident on consumer GPUs. But the abstraction must not
become DSV4-specific — OLMoE is the current correctness golden model and more
model families (Qwen MoE, Mixtral, …) will follow.

The current hot path for a DSV4 decode step looks like this:

```
router selects experts (per layer)
  → ExpertStreamingPlanner.plan_layer_step()
       produces loads[] + evictions[]
  → for each load:
       ExpertStreamingReader.read_load_source()   // synchronous byte-range read
       ExpertComputeBundle::from_artifact_payload()
       DeepSeekV4CudaOperatorCache.upload_expert_bundle()  // synchronous H2D
       experts.insert(expert_id, handles)
  → for each route:
       artifact_fp4_swiglu_ffn_matvec(gate, up, down, input, weight)
  → planner.commit_step()
```

The pain points, confirmed by reading the code:

1. **Disk read and H2D happen synchronously on the decode hot path.** A selected
   expert miss blocks the entire token until bytes are read from disk and
   uploaded to the GPU.
2. **No host-staged replica.** Every miss goes disk → host bytes → GPU. Even if
   the same expert was used 3 steps ago and evicted, it must be re-read from
   disk.
3. **Eviction is recency-only.** `ExpertState` has `last_used_step` but no
   `activation_count`. An expert selected 10 steps ago with high cumulative
   activation can be evicted in favor of one selected last step but rarely
   used. On skewed MoE workloads this thrashes.
4. **`ExpertLoadSource` conflates three concerns**: where bytes come from, what
   tier they are on, and whether they are resident. This makes it hard to add a
   host-staging tier without inventing new variants.
5. **Three parallel residency systems exist in code** (see [Current code
   reality](#current-code-reality)) and they are not coherent.
6. **No per-tier hit/miss counters.** It is impossible to tell whether a host
   cache would help, because the current counters only record loads/evictions
   without distinguishing disk-read vs host-hit vs device-resident-hit.

The storage abstraction must support DSV4's expert streaming, but it must remain
model-family agnostic so OLMoE, Qwen MoE, and future families plug in through
the same vocabulary.

---

## Goals and non-goals

### Goals

- **Stable object identity** that survives eviction, tier movement, and backend
  swaps. Identity must be content-addressable so host cache / disk cache /
  remote cache can all key on it.
- **Separate locators from replicas.** "Where can I fetch bytes from" and
  "where does a copy currently live" are different questions with different
  lifetimes. Conflating them is the root cause of pain point 4.
- **Host-staged expert cache as the first measurable win.** This is the
  highest-ROI change: it turns a disk-read miss into a host-memory hit without
  any async runtime.
- **Frequency-aware eviction.** Track `activation_count` alongside recency so
  hot experts survive budget pressure.
- **Integrate with runtime graph, CUDA, and WeightPack** without leaking
  backend details across boundaries.
- **Measurable counters** that can answer "did the host cache help?" and "is
  prefetch accurate?".

### Non-goals (for v1)

- **No async I/O.** Phase 0–2 stay synchronous. io_uring, pinned buffers, and
  async CUDA streams arrive in Phase 4, behind the same trait.
- **No remote cache.** RDMA / Mooncake-like / HiCache-like distributed caches
  are Phase 5 backend implementations, not v1 architecture.
- **No CUDA handles in graph IR.** Graph externals reference storage object IDs;
  backend lowering produces handles.
- **No model-family-specific storage APIs.** DSV4 expert layout, OLMoE expert
  layout, Qwen MoE layout — all become `StorageObjectDescriptor` metadata, not
  storage-layer variants.
- **No full mutable-state unification in v1.** KV pages and decode arena
  buffers share `Placement` vocabulary later, but are not forced through the
  locator/transfer path initially (see [V1 scope](#v1-scope)).
- **No rename-everything big bang.** Introduce vocabulary types with
  compatibility adapters; migrate call sites incrementally.

---

## Current code reality

Before designing the target, here is what actually exists in the codebase.
Two residency-related systems coexist and are not coherent:

### System 1: `ExpertStreamingPlanner` — active, expert-only

```
crates/ferrule-runtime/src/expert_streaming.rs
  ExpertStreamingPlanner, ExpertStreamingPolicy, ExpertStreamingStep
  ExpertLoadSource, ExpertStorageTier, ExpertLoadRequest, ExpertEvictRequest
```

This is the live planner. Used by:

- `deepseek_v4.rs` — DSV4 decode/prefill MoE step
- `layer_binding.rs` — generic layer binding
- `routed_moe.rs` — reference routed MoE execution
- `inspect.rs` (CLI) — expert inspection
- `tests/deepseek_streaming_local.rs`

It already supports: selected/predicted split, slot-based GPU residency, LRU
eviction via `last_used_step`, prefetch requests, and `commit_step`.

Its limitations (what the new design must fix without rewriting it wholesale):

| Limitation | Impact |
|---|---|
| Expert-only, not general object | Cannot reuse for artifact tensors, output-head rows, etc. without forking |
| `last_used_step` only, no `activation_count` | Eviction thrashes on skewed MoE (the F2 problem) |
| `commit_step` marks loads as `Gpu` unconditionally | No transfer-failure / partial-completion state |
| No host-staged replica tier | Every miss = disk read |
| No per-tier hit/miss counters | Cannot measure if host cache helps |
| `ExpertLoadSource` conflates source + tier + residency | Adding a host tier requires new variants |

### System 2: `DeepSeekV4CudaOperatorCache` — active, inline

```
crates/ferrule-runtime/src/models/deepseek_v4.rs  (L1300+)
  DeepSeekV4CudaOperatorCache {
    experts: HashMap<ExpertId, CudaFp4ExpertHandles>,
    expert_selected / expert_loads / expert_load_bytes / expert_evictions: u64
  }
```

This is the CUDA device-side expert cache for DSV4. Its `routed_moe_step`
(L1669) does the full inline loop:

```
planner.plan_layer_step()           → streaming.loads[] + evictions[]
handles.apply_evictions()           → remove from CPU handle store
self.experts.remove()               → remove from CUDA handle map
for load in loads:
  reader.read_load_source()         → synchronous disk read
  upload_expert_bundle()            → synchronous H2D
  self.experts.insert()
  handles.insert_resident_handle()  → mark Gpu in CPU store
for route in routes:
  self.experts.get()                → execute
planner.commit_step()
```

This is the real hot path. It already has per-expert device handles (not a
concatenated buffer), per-expert eviction, and load/byte/eviction counters. It
is the natural integration point — **not** something to rewrite.

Note: the OLMoE legacy CUDA path (`ferrule-cuda/build.rs` + `transformer/moe.rs`)
uses a different layout — all experts concatenated into one `ex_gate_packed`
`DeviceBuffer` with `ExpertQuantOffsets` addressing. That path is all-resident,
no eviction, no streaming. It is **not deprecated** — OLMoE remains the
correctness golden model and future models may use all-resident mode when they
fit. The storage abstraction must accommodate both layouts.

### Supporting types already in place

| Type | File | Role |
|---|---|---|
| `ArtifactTensorSlice` | `artifact_tensor.rs` | Byte-range artifact descriptor (path/offset/bytes/dtype/shape) |
| `ArtifactTensorReader` | `artifact_tensor.rs` | Generic file reader for artifact slices |
| `ExpertTensorSlice` | `expert_streaming.rs` | Expert-specific slice (key + component + path + offset) |
| `ExpertHandleStore` (trait) | `expert_handle.rs` | Backend-agnostic resident handle registry |
| `CpuExpertHandleStore` | `expert_handle.rs` | CPU `BTreeMap` impl (only impl today) |
| `ResidentExpertHandle` | `expert_handle.rs` | Resident handle record (expert/tier/format/bytes/slot) |
| `WeightPackReader` / `WeightPackSlice` | `ferrule-cuda/weightpack.rs` | mmap'd WeightPack package reader, zero-copy slices |
| `BackendObjectStore` | `backend_object_store.rs` | Graph external → concrete object materialization |
| `ExternalBindingKind` | `graph_runtime.rs` | Graph-side external binding enum (Weight/KvState/ArtifactTensor/ExpertRegistry/ResidentExpert/…) |
| `TransferCounters` / `ExpertRuntimeCounters` | `ferrule-bench/summary.rs` | Bench-side H2D/expert counters |

### Naming conflict: `ResidencyPolicy`

`ferrule-model/src/support/policies.rs` already defines:

```rust
pub struct ResidencyPolicy {
    pub streaming_allowed: bool,
    pub all_resident_required: bool,
}
```

This is a **model-level** policy (does this model require all experts resident,
or is streaming allowed?). The storage layer's policy is a different thing
(per-tier budgets, prefetch window, eviction). To avoid collision, the storage
layer's policy type is named `StorageResidencyPolicy` in this document.

```
Model-level ResidencyPolicy (ferrule-model):
  all_resident vs streaming_allowed

Runtime-level StorageResidencyPolicy (this design):
  budgets, prefetch window, retain_hot, eviction weights
```

---

## Unifying the two systems

The two active systems are **not competitors** — they are different layers that
happen to both track residency state:

```
┌─────────────────────────────────────────────────────────────────┐
│  STRATEGY LAYER                                                 │
│  ExpertStreamingPlanner (expert_streaming.rs)                   │
│    input:  selected[], predicted[], policy                      │
│    output: ExpertStreamingStep { loads[], evictions[] }         │
│    state:  BTreeMap<ExpertId, ExpertState {                     │
│               load_source, location, last_used_step             │
│             }>                                                  │
│    owns:   NO backend handles                                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │ ExpertStreamingStep
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  EXECUTION LAYER                                                │
│  DeepSeekV4CudaOperatorCache (deepseek_v4.rs)   ← CUDA path     │
│  CpuExpertHandleStore        (expert_handle.rs) ← CPU path      │
│    input:  ExpertStreamingStep                                  │
│    action: apply evictions → read → upload → insert             │
│    state:  HashMap<ExpertId, CudaFp4ExpertHandles>  (CUDA)      │
│             BTreeMap<ExpertId, ExpertComputeHandle> (CPU)       │
│    owns:   backend handles (CUDA buffers / CPU payloads)        │
└─────────────────────────────────────────────────────────────────┘
```

The split is sound. The problem is that the **execution loop is duplicated**:

```rust
// DeepSeekV4CudaOperatorCache::routed_moe_step (L1693-1712) — CUDA
handles.apply_evictions(&streaming.evictions);
for eviction in &streaming.evictions { self.experts.remove(&eviction.expert); }
for load in &streaming.loads {
    let payload = reader.read_load_source(...)?;
    let expert = self.upload_expert_bundle(&bundle)?;
    self.experts.insert(load.expert, expert);
    handles.insert_resident_handle(...)?;
}

// execute_routed_moe_reference_with_handles (routed_moe.rs L79-83) — CPU
handles.apply_evictions(&streaming.evictions);
for load in &streaming.loads {
    let payload = reader.read_load_source(...)?;
    handles.insert_artifact_payload(payload)?;
}
```

The CUDA version adds `upload_expert_bundle` + `self.experts.insert`, but the
structure is identical. Every new backend would copy-paste this loop.

### Unification path: extract a trait, not merge the systems

The right move is **not** to merge planner + executor into one struct. That
would couple strategy to backend. Instead, extract the common execution loop
into a trait that both backends implement:

```rust
/// Backend that can load, evict, and execute resident experts.
///
/// The planner produces an ExpertStreamingStep; the backend applies it.
/// This replaces the duplicated loop in routed_moe.rs / deepseek_v4.rs.
pub trait ExpertResidencyBackend {
    /// Remove an expert's backend handle (eviction).
    fn evict(&mut self, expert: ExpertId) -> Result<()>;

    /// Load bytes from source, upload to backend, return the handle.
    /// For CPU: stores the payload. For CUDA: uploads to device buffer.
    fn load_and_install(
        &mut self,
        expert: ExpertId,
        source: &ExpertLoadSource,
        reader: &ExpertStreamingReader,
    ) -> Result<u64>;  // returns bytes loaded

    /// Check if expert is currently resident on this backend.
    fn is_resident(&self, expert: ExpertId) -> bool;

    /// Number of currently resident experts.
    fn resident_count(&self) -> usize;

    /// Total bytes of resident handles.
    fn resident_bytes(&self) -> u64;
}
```

Then the common loop becomes one function:

```rust
pub fn apply_streaming_step(
    backend: &mut impl ExpertResidencyBackend,
    step: &ExpertStreamingStep,
    reader: &ExpertStreamingReader,
) -> Result<()> {
    for eviction in &step.evictions {
        backend.evict(eviction.expert)?;
    }
    for load in &step.loads {
        if !backend.is_resident(load.expert) {
            backend.load_and_install(load.expert, &load.load_source, reader)?;
        }
    }
    Ok(())
}
```

### Migration sequence

```
Phase 0 (now):
  ExpertStreamingPlanner          ExpertStreamingPlanner
  + CpuExpertHandleStore    →     + CpuExpertHandleStore
  + DeepSeekV4CudaOperatorCache   + CudaExpertResidencyBackend (new, wraps cache)
  (duplicated loops)              + apply_streaming_step() (one loop)

Phase 1:
  ExpertState gains activation_count
  ExpertStreamingPolicy gains StorageResidencyPolicy weights
  (planner augmented, not replaced)

Phase 2:
  HostStagedExpertCache becomes a third ExpertResidencyBackend
  (host tier, between disk and device)
  apply_streaming_step works for host→device without code changes

Phase 3+:
  ExpertId → StorageObjectId (key migration)
  ExpertStreamingPlanner → generalizes to non-expert objects
  ExpertResidencyBackend → ResidencyManager + TransferEngine split
```

### Why not merge into one struct now

1. **Strategy and execution have different lifetimes.** The planner survives
   across backends (CPU test, CUDA run). The backend is swapped per device. If
   merged, you cannot test the planner without a backend.
2. **The planner must stay backend-agnostic.** It lives in `ferrule-runtime`
   and is used by CPU reference tests. CUDA types must not leak into it.
3. **`ExpertHandleStore` already tried to be the unified interface** but only
   covers the handle registry, not the load/upload/evict loop. The missing
   piece is `ExpertResidencyBackend`, not a bigger planner.

### What changes in the code now

The CUDA path's inline loop (`deepseek_v4.rs` L1693-1712) and the CPU path's
loop (`routed_moe.rs` L79-83) both call `apply_streaming_step`. The CUDA cache
implements `ExpertResidencyBackend` by wrapping its existing `experts` HashMap.
The CPU store implements it by wrapping its existing `handles` BTreeMap. No
behavior change — just de-duplication.

---

## Target vocabulary

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STORAGE OBJECT                              │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ StorageObjectId│  │  Descriptor  │  │   ObjectLocator(s)       │ │
│  │ (enum identity) │  │ kind, bytes, │  │ LocalFile / LocalMmap /  │ │
│  │ content-addr    │  │ layout,      │  │ WeightPack / RemoteObj / │ │
│  │                │  │ mutability   │  │ RemoteCache              │ │
│  └───────────────┘  └──────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
         │                                            │
         │  (catalog maps ID → descriptor + locators) │
         ▼                                            ▼
┌─────────────────────────┐              ┌────────────────────────────┐
│    ResidencyManager     │              │      TransferEngine        │
│  ┌───────────────────┐  │              │  ensure()  → blocking      │
│  │  ObjectReplica[]  │  │  miss →      │  prefetch()→ async tickets │
│  │  placement, state,│  │  transfer    │  poll()    → events        │
│  │  generation,      │  │              │                            │
│  │  ReplicaHandleId  │  │              │  backends:                 │
│  └───────────────────┘  │              │  - file read / mmap        │
│  budgets (bytes+slots)  │              │  - WeightPack slice        │
│  hit/miss per tier      │              │  - H2D / D2H               │
└─────────────────────────┘              │  - (future) io_uring/RDMA  │
         ▲                               └────────────────────────────┘
         │                                           ▲
         │  (policy ranks objects, requests residency)│
         ▼                                           │
┌─────────────────────────────────────────────────────────────────────┐
│                            POLICY                                   │
│  StorageResidencyPolicy { budgets, retain_hot, prefetch_window }    │
│  ResidencyScore { execute_now, predicted, recency, freq, cost }     │
│  ranks → ResidencyRequest[]                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Storage object identity

Identity must be stable across eviction, tier movement, and backend swaps, and
must be content-addressable so that host cache / disk cache / remote cache can
all key on it reliably. A bare `String` newtype is not enough: it loses kind
information at the type level, forces every lookup through a HashMap, and
makes debugging opaque. The ID is therefore a structured enum whose variants
match `StorageObjectKind`, with a `Display` impl that produces the canonical
content-addressed string for cache keys.

```rust
/// Logical identity for a loadable/resident runtime object.
///
/// Structured by design: the variant IS the kind, so the catalog can index by
/// `(kind, structural key)` without a separate descriptor lookup, and debug
/// output shows what the object is without decoding a string.
pub enum StorageObjectId {
    ArtifactTensor {
        model_revision: ModelRevision,
        tensor_role: TensorRole,
        dtype: ArtifactDType,
        shape_hash: u64,
    },
    ArtifactTensorRows {
        model_revision: ModelRevision,
        tensor_role: TensorRole,
        row_start: u64,
        row_end: u64,
        dtype: ArtifactDType,
    },
    ExpertBundle {
        model_revision: ModelRevision,
        layer: u32,
        expert: u32,
        layout_version: u32,
    },
    ExpertMatrix {
        model_revision: ModelRevision,
        layer: u32,
        expert: u32,
        matrix_kind: ExpertMatrixKind,   // Gate / Up / Down
        component: ExpertTensorComponent, // Weight / Scale / Other
    },
    OutputHeadChunk {
        model_revision: ModelRevision,
        chunk: u32,
        dtype: ArtifactDType,
    },
    WeightPackChunk {
        weightpack_id: WeightPackId,
        chunk_id: String,
        layout_version: u32,
    },
    // --- v2+ (not in v1 catalog, but Placement-compatible) ---
    KvPage { session: SessionId, page: u64 },
    DecodeArenaBuffer { device_id: u32, slot: u32 },
    Opaque { tag: String, key: String },
}

/// Content-addressed model identity. Derived from config hash + tokenizer hash
/// + quant policy. Two models with different revisions never share objects.
pub struct ModelRevision(pub u64);

pub struct WeightPackId(pub u64);  // hash of WeightPack manifest

// --- Auxiliary types referenced by StorageObjectId variants ---
// These already exist or have clear analogues in the codebase:
//   ExpertMatrixKind  → expert_streaming.rs (Gate/Up/Down)
//   ExpertTensorComponent → expert_streaming.rs (Weight/Scale/Other)
//   ArtifactDType     → artifact_tensor.rs (F32/BF16/F4E2M1/...)
//   TensorRole        → model-family binding metadata (e.g. "q_proj", "gate")
//   SessionId         → session.rs
// They are reused, not redefined, when the storage vocabulary lands.
pub struct TensorRole(pub String);  // semantic role tag, e.g. "q_proj", "expert.gate"
```

The `Display` impl produces the canonical cache-key strings used by remote
and disk caches:

```
artifact:{rev}/tensor/{role}/{dtype}/{shape_hash}
artifact:{rev}/tensor/{role}/rows/{start}:{end}/{dtype}
expert:{rev}/layer{L}/expert{E}/bundle/v{layout_version}
expert:{rev}/layer{L}/expert{E}/{matrix_kind}/{component}
output_head:{rev}/chunk{C}/{dtype}
weightpack:{wp_id}/chunk/{chunk_id}/v{layout_version}
```

`model_revision` comes from the model config hash / WeightPack manifest.
`layout_version` comes from the WeightPack format version or artifact format
tag. This makes cache invalidation automatic: a different quantization, a
different layout version, or a different model produces a different ID —
whether compared as an enum or as its canonical string.

**Why an enum, not a String newtype:**

| Concern | `StorageObjectId(String)` | `StorageObjectId` enum |
|---|---|---|
| Kind at type level | no — must query descriptor | yes — variant IS the kind |
| Catalog index | `HashMap<String, …>` | `HashMap<(Kind, structural_key), …>` or per-variant maps |
| Debug output | opaque string | variant + fields |
| Invalid free-form strings | cannot reject | impossible by construction |
| Cache-key serialization | the value itself | `Display` / `to_string()` |
| Match on kind in policy | impossible without lookup | `match id { ExpertBundle { .. } => … }` |

### Storage object descriptor

```rust
pub enum StorageObjectKind {
    ArtifactTensor,
    ArtifactTensorRows,
    ExpertMatrix,
    ExpertBundle,
    WeightPackChunk,
    // --- v1 scope ends here ---
    KvPage,              // v2+ — shares Placement vocab, not locator/transfer
    DecodeArenaBuffer,   // v2+ — backend memory planning, not a fetched object
    GraphExternal,
    Opaque,
}

pub enum StorageMutability {
    Immutable,           // expert weights, artifact tensors, WeightPack chunks
    Mutable,             // KV pages (future)
    Ephemeral,           // decode arena buffers (future — no locator)
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

Layout is semantic enough for validation and backend selection, but not
model-family-specific. DSV4 "layer 17 expert 42" is model-binding metadata;
storage only sees an expert bundle object with size/layout/locators.

```rust
pub enum StorageLayout {
    Bytes,
    Tensor(TensorLayout),
    ExpertBundle(ExpertBundleLayout),
    KvPage(KvPageLayout),       // v2+
    Opaque { tag: String },
}

// Sketched in appendix-level detail when first implemented; not v1-blocking.
pub struct TensorLayout { /* dtype, shape, stride */ }
pub struct ExpertBundleLayout { /* gate/up/down shapes, quant format */ }
pub struct KvPageLayout { /* head_dim, num_heads, page_size */ }
```

### Placement

Placement describes where a replica **currently is**, not where it came from.

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

/// Phase 5 — sketched now so Placement is complete. Not constructed in v1.
pub struct RemoteEndpoint {
    pub scheme: RemoteScheme,
    pub host: String,
    pub port: Option<u16>,
}
pub enum RemoteScheme {
    Rdma,       // Mooncake-like distributed cache
    Http,       // object store / S3-compatible
    Grpc,       // LAN cache service
}
```

**Tier hierarchy** (conceptual; backends may skip edges):

```
  Remote  ──────────────────── (Phase 5: RDMA / object store / LAN cache)
     │
     ▼  fetch / cache
  Disk    ──────────────────── safetensors / WeightPack / local cache log
     │
     ▼  read / mmap / io_uring (Phase 4)
  Host    ──────────────────── staged bytes (pinned or unpinned)
     │
     ▼  H2D / async copy stream (Phase 4+)
  Device  ──────────────────── CUDA handles / device buffers
```

### Locator

A locator says how an object can be fetched if no suitable replica is resident.
Locators are catalog entries — not execution handles.

```rust
pub enum ObjectLocator {
    LocalFile { path: PathBuf, offset: u64, bytes: u64 },
    LocalMmap { path: PathBuf, offset: u64, bytes: u64 },
    WeightPack { path: PathBuf, chunk: String, offset: u64, bytes: u64 },
    RemoteObject { uri: String, offset: u64, bytes: u64 },
    RemoteCache { key: String, offset: u64, bytes: u64 },  // offset added for range parity
}
```

`RemoteCache` now carries `offset` for range-based fetch parity with
`RemoteObject`. If a backend only supports whole-object caching, it ignores
`offset`.

### Replica and handle

A replica is a current resident copy. The generic manager tracks **metadata
only** — real backend handles (CUDA buffers, mmap slices) stay in backend-owned
stores.

```rust
pub struct ObjectReplica {
    pub object: StorageObjectId,
    pub placement: Placement,
    pub bytes: u64,
    pub state: ReplicaState,
    pub generation: u64,
    pub handle: ReplicaHandleId,   // opaque reference to backend-owned handle
}

pub enum ReplicaState {
    Ready,
    Loading,
    Evicting,
    Failed { reason: String },
}

/// Opaque handle reference. The generic manager never dereferences this;
/// the backend that owns the handle (e.g. DeepSeekV4CudaOperatorCache.experts)
/// resolves it. This keeps CUDA types out of the runtime crate.
pub struct ReplicaHandleId {
    pub backend: BackendId,
    pub slot: u64,
    pub generation: u64,
}

pub type BackendId = String;  // "cuda:0", "cpu", "host-staged-cache", ...
```

**Why opaque handles:** Today `DeepSeekV4CudaOperatorCache.experts` is
`HashMap<ExpertId, CudaFp4ExpertHandles>`. The new design does **not** move
`CudaFp4ExpertHandles` into `ObjectReplica` — that would pull CUDA types into
the runtime crate. Instead, the residency manager records a `ReplicaHandleId`,
and `DeepSeekV4CudaOperatorCache` continues to own the real handles. Over time
the key changes from `ExpertId` to `StorageObjectId`:

```
experts: HashMap<ExpertId, CudaFp4ExpertHandles>       // today
experts: HashMap<StorageObjectId, CudaFp4ExpertHandles> // after migration
```

### Residency request and reason

```rust
pub struct ResidencyRequest {
    pub object: StorageObjectId,
    pub desired: Placement,               // precise, not just a class
    pub priority: ResidencyPriority,
    pub deadline: Option<Instant>,
    pub reason: ResidencyReason,
}

pub enum ResidencyPriority {
    Critical,   // selected expert, blocks current token
    High,       // predicted expert, should be ready soon
    Low,        // retain-hot, keep if budget allows
    Background, // prefetch, best-effort
}

pub enum ResidencyReason {
    ExecuteNow,   // selected by router, must be device-resident now
    Prefetch,     // predicted, stage at host or device
    RetainHot,    // not needed now but high cumulative frequency — keep if possible
    Debug,        // diagnostic / forced residency
}
```

`ResidencyReason::ExecuteNow` implies `ResidencyPriority::Critical`.
`desired` uses precise `Placement` (not a coarse class) so the manager does not
have to guess which `device_id` or `memory` kind — the caller knows.

**No `PlacementClass` enum.** An earlier draft introduced a coarse
`PlacementClass { Device, Host, LocalDisk, Remote }` alongside the precise
`Placement`. That created an undefined resolution gap: the manager would
receive "Device" but still have to pick `device_id` and `memory` kind itself
— a latent bug source for multi-GPU. The current design uses `Placement`
directly in `ResidencyRequest.desired`. If a caller only cares about the tier
and not the specific device, it constructs `Placement::Local(LocalPlacement::
Device { device_id: 0, memory: DeviceMemoryKind::Vram })` from context — the
decision is explicit at the call site, not hidden in the manager.

---

## Component boundaries

### 1. Storage catalog

**Owns** object descriptors and locators.

Responsibilities:

- map semantic artifacts to storage objects,
- register local and remote locators,
- normalize safetensors / WeightPack / cache chunk metadata,
- answer "where can object X be fetched from?"

Non-responsibilities: no scheduling, no prefetch decisions, no CUDA handles.

```rust
pub trait StorageCatalog {
    fn descriptor(&self, id: &StorageObjectId) -> Option<&StorageObjectDescriptor>;
    fn locators(&self, id: &StorageObjectId) -> &[ObjectLocator];
}
```

**Relationship to `BackendObjectStore`:** `BackendObjectStore` is the current
graph-external materialization layer. `StorageCatalog` is **not** a parallel
competitor — it is the next layer below `BackendObjectStore`. The integration
stack is:

```
ExternalBindingPlan
  → BackendObjectStore          (graph → concrete object materialization)
    → StorageCatalog            (object ID → descriptor + locators)
      → ResidencyManager        (object ID → current replicas)
        → TransferEngine        (execute movement)
          → Backend handle store (CUDA/mmap/pinned — backend-owned)
```

### 2. Residency manager

**Owns** current replicas and budgets.

Responsibilities:

- answer "is object X available at placement Y?",
- track host/device/disk/remote replicas,
- enforce budgets (**both bytes and slots** — see below),
- choose evictions when policy asks for space,
- expose telemetry (hit/miss per tier, bytes transferred, replica lifetimes).

```rust
pub struct ResidencyManager {
    catalog: Box<dyn StorageCatalog>,
    replicas: HashMap<StorageObjectId, Vec<ObjectReplica>>,
    budgets: Budgets,
    counters: ResidencyCounters,
}

pub struct Budgets {
    pub device_slots_per_layer: usize,   // correctness: >= num_experts_per_tok
    pub device_budget_bytes: u64,
    pub host_staging_budget_bytes: u64,
    pub local_disk_cache_budget_bytes: Option<u64>,
}
```

**Why both bytes and slots:** DSV4 routed experts are roughly uniform in size,
but different layouts/quant/roles are not guaranteed to be. Per-layer top-k
correctness requires at least `num_experts_per_tok` device slots — a pure
byte budget can violate this. Host-staged cache is better suited to bytes.
The policy carries both.

### 3. Transfer engine

**Executes** movement plans.

```rust
pub trait TransferEngine {
    /// Blocking ensure: make the object available at the requested placement.
    /// Returns a handle id the caller can resolve with the backend.
    fn ensure(&mut self, request: ResidencyRequest) -> Result<ReplicaHandleId>;

    /// Non-blocking prefetch: start transfers, return tickets for polling.
    /// Phase 4+; v1 implementation may panic or fall back to ensure.
    fn prefetch(&mut self, requests: &[ResidencyRequest]) -> Result<Vec<TransferTicket>>;

    /// Poll for completion of previously issued transfers.
    fn poll(&mut self) -> Result<Vec<TransferEvent>>;
}

pub struct TransferTicket { pub id: u64, pub object: StorageObjectId }
pub struct TransferEvent {
    pub ticket: u64,
    pub object: StorageObjectId,
    pub outcome: TransferOutcome,
}
pub enum TransferOutcome {
    Completed(ReplicaHandleId),
    Failed { reason: String },
    Cancelled,
}
```

The first implementation is synchronous and boring: `ensure` does a file read +
optional H2D, `prefetch`/`poll` are stubs. io_uring, pinned memory, RDMA, and
async CUDA streams arrive in Phase 4 behind the same trait — planner semantics
do not change.

### 4. Policy / planner

**Decides** what should be resident or staged.

```rust
pub struct StorageResidencyPolicy {
    pub budgets: Budgets,
    pub retain_hot: bool,
    pub prefetch_window: usize,
    pub eviction_weights: EvictionWeights,
}

pub struct EvictionWeights {
    pub recency: f32,         // last_used_step
    pub frequency: f32,       // activation_count (F2 skew-aware)
    pub load_cost: f32,       // how expensive is a miss?
}

pub struct ResidencyScore {
    pub execute_now: bool,
    pub predicted: bool,
    pub last_used_step: u64,
    pub activation_count: u64,    // frequency counter
    pub load_cost_bytes: u64,
    pub object_bytes: u64,
}
```

The first policy is simple:

1. selected objects must be device-resident (`ExecuteNow → Device`),
2. predicted objects should be host-staged, or device-resident if space remains
   (`Prefetch → Host` or `Device`),
3. hot resident objects kept while budget allows (`RetainHot`, using **both**
   recency and frequency — evict coldest-by-both first),
4. evict coldest objects first (cold-by-recency **AND** cold-by-frequency).

No async priority queue is required for v1.

**Relationship to `ExpertStreamingPlanner`:** The planner is **not** rewritten.
Phase 1 augments it: add `activation_count` to `ExpertState`, route eviction
through `StorageResidencyPolicy` weights, and emit `ResidencyRequest`-shaped
output. It generalizes to other object kinds later, not in Phase 1.

---

## V1 scope

### What v1 manages

**Immutable loadable objects only:**

- `ArtifactTensor`, `ArtifactTensorRows`
- `ExpertMatrix`, `ExpertBundle`
- `WeightPackChunk`

These have stable bytes, locators, checksums, and sizes. They fit the
locator/transfer model naturally.

### What v1 does NOT manage

**Mutable execution state** — `KvPage`, `DecodeArenaBuffer`, graph-external
runtime state — shares `Placement` vocabulary later, but is not forced through
the locator/transfer path in v1.

`DecodeArenaBuffer` in particular is a backend memory-planning artifact, not a
fetched object. It has no locator. Calling it a storage object would force
`ObjectLocator` to grow a "no locator" variant, which muddies the model. It
enters the vocabulary in v2+ as a `Placement`-tagged object only.

### v1 expert policy has both slots and bytes

```
per-layer minimum selected slots   (correctness: >= num_experts_per_tok)
per-tier byte budget               (capacity: device / host / disk)
```

Do not drop slots in favor of bytes — the current `ExpertStreamingPolicy`
`gpu_slots_per_layer` is a correctness constraint, not just a capacity hint.

### Expert bundles are the first ROI target — within storage/residency

This does **not** reorder the broader DSV4 device-resident decode roadmap. The
DSV4 performance priority sequence (decode arena, device-resident KV, artifact
linear dispatch, expert residency, output-head/top-k) is tracked in
`TEMP_INFERENCE_ROADMAP.md`. Expert residency is the first ROI target **inside
the storage/residency layer**, not the first global DSV4 task.

---

## Expert residency flow

### Current hot path (DSV4, as in code today)

```
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌─────────┐
│ router  │───▶│ planner  │───▶│ sync disk │───▶│ sync H2D │───▶│ execute │
│ top-k   │    │ plan_    │    │ read      │    │ upload   │    │ matvec  │
│ select  │    │ layer_   │    │ (miss)    │    │          │    │         │
│         │    │ step     │    │           │    │          │    │         │
└─────────┘    └──────────┘    └───────────┘    └──────────┘    └─────────┘
                      │
                      ▼
                 evictions[] ──▶ remove from device + handle store
```

Every miss = disk read + H2D, synchronously on the decode path.

### Target v1 flow (synchronous, host-staged)

```
┌─────────┐    ┌──────────┐    ┌──────────────────────────────────────────┐
│ router  │───▶│ planner  │───▶│ for each load:                           │
│ top-k   │    │ plan_    │    │   host cache hit?                        │
│ select  │    │ layer_   │    │     yes ─▶ decode/upload ─▶ execute      │
│         │    │ step     │    │     no  ─▶ disk read ─▶ insert host cache │
└─────────┘    └──────────┘    │            ─▶ upload ─▶ execute          │
                      │        └──────────────────────────────────────────┘
                      ▼
                 evictions[] ──▶ remove device replica (host replica stays!)
```

Key change: a host-staged replica survives device eviction. A re-selected
expert that was evicted 3 steps ago hits host cache instead of re-reading disk.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RESIDENCY TIERS (v1)                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────┐  miss  ┌─────────────────┐ │
│  │ DEVICE (CUDA handles)               │◀───────│ HOST STAGED     │ │
│  │ experts: HashMap<Id, CudaFp4Handles>│        │ (Arc<[u8]>)     │ │
│  │ budget: slots + bytes               │        │ budget: bytes   │ │
│  │ eviction: freq + recency            │        │ eviction: LRU   │ │
│  └─────────────────────────────────────┘        └────────┬────────┘ │
│       ▲ miss                                              │ miss     │
│       │                                                   ▼          │
│       │           ┌────────────────────────────────────────┐         │
│       └───────────│ DISK (locators)                        │◀────────┘
│                   │ LocalFile / LocalMmap / WeightPack     │
│                   │ (no budget in v1; local cache = v2+)   │
│                   └────────────────────────────────────────┘
└──────────────────────────────────────────────────────────────────────┘
```

### Target v2 flow (async prefetch overlap)

```
previous step prediction
  → background prefetch reads predicted experts into host pinned memory
current step selection
  → selected expert usually host-staged or device-resident
  → H2D upload overlaps on a transfer stream
  → compute stream waits only when needed
```

### Target v3 flow (remote/disk cache hierarchy)

```
remote/disk cache hierarchy
  → local disk cache (append-only log + circular buffer)
  → host staging
  → async H2D
  → device-resident execution
```

---

## Integration points

### Runtime graph

Graph externals reference **storage object IDs**, never concrete backend
handles. The `ExternalBindingKind` enum already exists in `graph_runtime.rs`:

```
ExternalBindingKind::ArtifactTensor  → StorageObjectId
ExternalBindingKind::ExpertRegistry  → collection of StorageObjectId
ExternalBindingKind::KvState         → mutable storage object / state handle (v2+)
ExternalBindingKind::ResidentExpert  → device placement requirement, not CUDA handle
```

Backend lowering turns object IDs and placement requirements into actual
handles. `BackendObjectStore` is the current materialization layer; it gains a
`StorageCatalog` dependency rather than being replaced by one.

### CUDA

`ferrule-cuda` exposes device-handle primitives and transfer counters, but does
not own model-family storage policy.

CUDA responsibilities (unchanged):

- upload host bytes to device buffers / handles,
- later support pinned buffers and async copy streams,
- expose device replicas as opaque handles,
- report H2D/D2H bytes, copy counts, kernel launches.

Runtime storage responsibilities:

- decide what to stage / retain / evict,
- choose local vs remote locator,
- connect object IDs to CUDA uploads.

`DeepSeekV4CudaOperatorCache` keeps owning `experts: HashMap<…,
CudaFp4ExpertHandles>`. The key migrates from `ExpertId` to `StorageObjectId`.
The residency manager references these via `ReplicaHandleId`, never by
directly holding `CudaFp4ExpertHandles`.

### WeightPack

`WeightPackReader` is currently mmap + layer offsets. The residency layer needs
object-level chunk metadata, not just layer offsets.

**WeightPack vNext requirements:**

```
- object table keyed by StorageObjectId
- per-chunk offset / bytes / checksum
- layout version
- per-role quant policy
- (optional) residency hints
```

This makes `ObjectLocator::WeightPack { chunk, offset, bytes }` resolvable from
the manifest rather than computed ad hoc. Existing `WeightPackManifest` fields
(`format_version`, `quant_type`, `layout_version`, `model_config_hash`) provide
the `model_revision` and `layout_version` inputs to `StorageObjectId`
generation.

### Model-level residency policy

`ferrule-model`'s `ResidencyPolicy { streaming_allowed,
all_resident_required }` decides whether a model **requires** all-resident
experts or allows streaming. The storage layer's `StorageResidencyPolicy`
decides **how much** to keep on each tier and what to evict. They compose:

```
ModelSupportContract / EnginePlan:
  ResidencyPolicy.streaming_allowed = false  → all experts loaded at startup, no eviction
  ResidencyPolicy.streaming_allowed = true   → StorageResidencyPolicy governs tiers

StorageResidencyPolicy (active only when streaming_allowed = true):
  budgets, retain_hot, prefetch_window, eviction weights
```

This accommodates both OLMoE (all-resident when it fits) and DSV4 (streaming
under memory pressure) under one framework.

---

## Metrics and validation

### Existing counters

`DeepSeekV4CudaOperatorCache` already records:

- `expert_selected`, `expert_loads`, `expert_load_bytes`, `expert_evictions`

`ferrule-bench` has `TransferCounters` (H2D/D2H bytes/copies) and
`ExpertRuntimeCounters` (loads/load_bytes/evictions/selected/resident).

### Counters needed for v1 (Phase 2 acceptance gate)

```
disk_read_bytes
disk_read_count
host_stage_hits
host_stage_misses
device_resident_hits
device_resident_misses
host_stage_bytes
host_stage_evictions
prefetch_issued          (v2+)
prefetch_completed       (v2+)
prefetch_used            (v2+ — prefetch that was actually selected)
prefetch_wasted          (v2+ — prefetched but evicted before use)
planner_cache_inconsistencies   (planner says Gpu but CUDA handle missing)
```

Without these, it is impossible to tell whether the host cache helps. Phase 2 is
not done until these counters exist and are wired into the JSON/runtime report.

---

## Implementation plan

### Phase 0 — vocabulary, trait extraction, cleanup

**No execution behavior change.** Introduce types, de-duplicate the execution
loop, clean up dead code.

- [ ] Add `StorageObjectId` (structured enum, not String newtype — see
      vocabulary section), `StorageObjectDescriptor`, `ObjectLocator`,
      `Placement`, `ObjectReplica`, `ReplicaHandleId`, `ResidencyRequest`,
      `ResidencyPriority`, `StorageResidencyPolicy`, `ResidencyScore`
      (with `activation_count` placeholder).
      Add `Display` for `StorageObjectId` producing canonical cache-key strings.
      Add `ModelRevision` / `WeightPackId` newtypes for content addressing.
- [ ] Name the policy `StorageResidencyPolicy` — **not** `ResidencyPolicy` — to
      avoid collision with `ferrule-model`.
- [ ] Extract `ExpertResidencyBackend` trait (see [Unifying the two
      systems](#unifying-the-two-systems)). Implement for
      `CpuExpertHandleStore` and `DeepSeekV4CudaOperatorCache`.
- [ ] Extract `apply_streaming_step(backend, step, reader)` — one loop
      replacing the duplicated loops in `routed_moe.rs` and `deepseek_v4.rs`.
- [ ] Add compatibility adapters: `ExpertId → StorageObjectId`,
      `ExpertLoadSource → ObjectLocator`, `ExpertStorageTier → Placement`.
- [ ] Do **not** rename call sites yet.

### Phase 1 — expert residency augmentation (DSV4 path, no rewrite)

**Keep `ExpertStreamingPlanner`. Augment, do not replace.**

- [ ] Add `activation_count: u64` to `ExpertState`.
- [ ] Route eviction through `StorageResidencyPolicy` weights (recency +
      frequency), replacing pure-LRU.
- [ ] Emit `ResidencyRequest`-shaped output from the planner (adapter, not new
      planner).
- [ ] Add planner/cache consistency check:
      `planner says Gpu` but `cuda experts missing` → treat as miss, repair
      planner state, count inconsistency.
- [ ] Do **not** touch the OLMoE legacy CUDA path (`build.rs` + `moe.rs`).
      All-resident mode stays for models that fit.
- [ ] Counters: `device_resident_hits`, `device_resident_misses`,
      `planner_cache_inconsistencies`.

### Phase 2 — host-staged expert cache

The first measurable win. No async runtime required.

- [ ] Add `HostStagedExpertCache` (small, concrete struct, not a trait yet):

```rust
pub struct HostStagedExpertCache {
    entries: LinkedHashMap<StorageObjectId, Arc<[u8]>>,
    budget_bytes: u64,
    current_bytes: u64,
    counters: HostCacheCounters,
}
```

- [ ] Read path becomes:
  ```
  host cache hit?  → decode bundle / upload → execute
  host cache miss? → disk read → insert host cache → upload → execute
  ```
- [ ] Device eviction keeps host replica (only removes device replica).
- [ ] Wire all Phase 2 counters into runtime JSON report.
- [ ] **Acceptance:** `disk_read_count` drops when experts re-activate;
      `host_stage_hits` > 0; no correctness regression on DSV4 smoke test.

### Phase 3 — prediction inputs (conservative)

Feed cheap prediction hints, staged carefully.

- **v3.0 (Phase 3):** record activation histogram during decode. No active
  prefetch. `activation_count` + `last_used_step` improve eviction only.
- **v3.5 (Phase 3.5):** hash-router layers predict next-step selected experts
  from sampled token id. Stage at host (not device) — cheap, no correctness
  risk.
- **v4.0 (Phase 4):** score-top-k router exposes top-k + margin candidates.
  Prefetch into host pinned memory on a background thread.

What **not** to predict first:

- full future token sequences,
- deep future layer activations before computing preceding layers,
- remote-cache hit probabilities without counters,
- expensive learned predictors before simple LRU/top-k+margin signals are
  measured.

### Phase 4 — asynchronous backends

Add transfer backends behind the same `TransferEngine` trait:

- threaded local reader,
- io_uring local reader (Linux),
- pinned host staging,
- async H2D transfer stream.

**Acceptance:** no runtime/model API changes when switching transfer backend.

### Phase 5 — remote/cache systems

Add remote/cache locators and transfer backends:

- local disk cache (append-only log + circular buffer, FASTER-style),
- remote object store (HTTP range / S3 GET),
- Mooncake-like distributed cache (RDMA-based, cluster-wide),
- HiCache-like adaptive per-tier eviction,
- LMCache-like unified KV+weight cache (brings `KvPage` into the catalog).

**Acceptance:** remote systems are just locator/transfer implementations;
graph/model/operator code does not branch on remote backend names.

---

## Risks and invariants

| Risk | Mitigation |
|---|---|
| Free-form string IDs sneak into the catalog | `StorageObjectId` is an enum — arbitrary strings cannot be constructed; `Opaque { tag, key }` is the only string-backed variant, reserved for non-v1 objects |
| Generic `ObjectReplica` accidentally holds backend handles | `ReplicaHandleId` is opaque; backend types never appear in runtime crate |
| Pure byte budget violates top-k slot correctness | `Budgets` carries both `device_slots_per_layer` and `device_budget_bytes` |
| Host cache helps disk reads but hurts H2D (double copy) | Measure `disk_read_count` vs `host_stage_hits` vs H2D bytes in Phase 2; abort host cache if net negative |
| Planner says resident but backend handle missing | Phase 1 consistency check + `planner_cache_inconsistencies` counter; treat as miss and repair |
| Async prefetch introduces use-after-free on device handles | `generation` on `ObjectReplica` + `ReplicaHandleId`; execution code checks generation before use |
| Over-generalization slows v1 | v1 scope explicitly limited to immutable loadable objects; KV/arena deferred |

### Invariants

1. **Objects are logical; replicas are physical.** Eviction removes a replica,
   not the object. The descriptor and locators are unchanged.
2. **Locators are catalog entries, not execution handles.** Never dereference
   a locator on the hot path — use it to drive a transfer.
3. **Backend handles stay in backend-owned stores.** The runtime crate never
   imports `CudaFp4ExpertHandles` or similar.
4. **`StorageResidencyPolicy` is active only when model-level
   `ResidencyPolicy.streaming_allowed = true`.** All-resident mode bypasses
   eviction entirely.
5. **Model-family specifics stay outside the storage layer.** "DSV4 layer 17
   expert 42" is binding metadata; storage sees an expert bundle object with
   size/layout/locators.

---

## External systems mapping

This appendix collects how external systems inform Ferrule's design. It is a
single reference table, not inline commentary through the document.

### Summary table

| System | Lesson adopted | Ferrule decision | Phase |
|---|---|---|---|
| FASTER (SIGMOD 2018) | Decouple logical identity from physical location; generation counters for safe concurrent eviction | `StorageObjectId` decoupled from `Placement`/`ObjectLocator`; `ObjectReplica.generation` | 0 |
| F2 (arXiv 2305.01516) | Skewed workloads reward frequency-aware eviction over pure LRU | `ResidencyScore.activation_count`; evict cold-by-recency **and** cold-by-frequency | 1 |
| Mooncake | Prefill/decode boundary as prediction point; distributed cache from pooled cluster resources | Prefetch hook at prefill→decode transition; `RemotePlacement` + `ObjectLocator::RemoteCache` as future backend | 3, 5 |
| HiCache | Unified multi-level cache for KV and weights; adaptive per-tier eviction | `StorageObject` unifies expert/KV/artifact under one catalog (KV in v2+); per-tier eviction weights | 2, 5 |
| LMCache | KV and weights as peers in one cache hierarchy; pluggable backends | `TransferEngine` trait is the pluggable backend; locator variants select backend | 4, 5 |
| vLLM / PagedAttention | Explicit, schedulable memory blocks; do not hide KV memory behind ad-hoc vectors | `KvPage` as a `StorageObject` (v2+); paged KV allocator interacts with `Placement` | 5+ |
| SGLang | Prefix/reuse metadata matters as much as raw kernel speed | Prefix cache → `ObjectLocator::RemoteCache` / local cache in Phase 5 | 5 |
| FlexGen / PowerInfer / LLM-in-a-Flash | Out-of-core scheduling; hot/cold locality; flash-aware loading | Expert host staging first (Phase 2); disk cache log later (Phase 5) | 2, 5 |
| llama.cpp | mmap/local file layout and partial offload are practical but insufficient for dynamic MoE staging | `LocalMmap` locator; but dynamic expert staging needs more than mmap | 0, 2 |

### Concept ancestry

```
                 FASTER
                   │
     logical addressing, gen-based consistency
                   │
              ┌────┴────────────┬─────────────────┐
              │                 │                 │
            F2              Mooncake         HiCache / LMCache
              │                 │                 │
     frequency-aware      prefill/decode      unified KV+weight
     skewed hotness       distributed cache   multi-level cache
              │                 │                 │
              └────────┬────────┴─────────────────┘
                       │
                Ferrule Storage
                & Residency Layer
                       │
          ┌────────────┼────────────┐
          │            │            │
   StorageCatalog  ResidencyManager  TransferEngine
          │            │            │
     object IDs +   replicas +     blocking read,
     locators       budgets +      mmap, io_uring,
                    eviction       RDMA, H2D, HTTP
```

### What we deliberately do NOT adopt

- **FASTER's HybridLog single address space.** Ferrule objects are discrete,
  multi-replica, scattered across tiers — not a contiguous log. We borrow
  logical/physical decoupling and generation counters only.
- **FASTER's "mutable region" analogy for device memory.** Expert weights are
  immutable at runtime; the mutable/immutable region split does not map
  cleanly. Device memory is simply the hottest tier.
- **Mooncake's prefill/decode disaggregation across nodes.** Ferrule is
  single-machine first. The lesson is the prediction boundary, not the
  disaggregated topology.
- **Per-tier eviction policies as separate systems.** HiCache uses different
  policies per level; Ferrule uses one `StorageResidencyPolicy` with
  configurable weights, keeping the system simple for v1.

---

## References

1. **FASTER: A Concurrent Key-Value Store with In-Place Updates.**
   Chandramouli et al., SIGMOD 2018.
   https://www.microsoft.com/en-us/research/publication/faster/
2. **From FASTER to F2: Evolving Concurrent Key-Value Store Designs for Large
   Skewed Workloads.** Kanellis et al., arXiv 2305.01516, 2023.
   https://arxiv.org/abs/2305.01516
3. **Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving.**
   Qin et al. (Moonshot AI / Tsinghua).
   https://arxiv.org/abs/2407.00079
4. **HiCache: Hierarchical Cache for Large Language Model Inference.**
   (Emerging system; design pattern for multi-level LLM caches. Referenced as
   a design pattern; no formal publication cited here.)
5. **LMCache: A Shared Cache Layer for LLM Serving.**
   (Emerging system; unified KV cache + weight cache for cross-instance
   sharing. Referenced as a design pattern; no formal publication cited here.)
6. **vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention.**
   Kwon et al., SOSP 2023.
7. **SGLang: Efficient Execution of Structured Language Model Programs.**
   Zheng et al., NeURIPS 2024.
8. **FlexGen: High-Throughput Generative Inference of Large Language Models with
   a Single GPU.** Sheng et al., ICML 2023.
9. **PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU.**
   Song et al., arXiv 2312.12456.
10. **LLM in a flash: Efficient Large Language Model Inference with Limited
    Memory.** Alizadeh et al., arXiv 2312.11514.
