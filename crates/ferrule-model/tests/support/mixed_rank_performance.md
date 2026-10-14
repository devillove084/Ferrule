# Mixed rank host-wall profiling v1 (not a benchmark)

## Runtime entry

Call the existing owner before a forward:

```rust,ignore
operators.set_mixed_rank_performance(true)?;
```

The flag defaults to false. The tracing subscriber must ALSO enable DEBUG for
`ferrule_performance`. Both gates are required. This API does not install a
numeric diagnostic observer, change upload-proof reuse, or invoke a CUDA API.
Setting false between invocations disables it. No runtime/composition/CLI changes
are included here. Runtime may separately time outer dispatch/publication.

The existing `execute_mixed_rank_segment` and standard layer loop own the hooks.
Enabled mode retains one fixed-size aggregate allocation per invocation and
stack/Rc scoped phase guards; no per-kernel logging, CUDA event, new fence,
readback, checkpoint access or driver-memory query is introduced. Disabled mode
returns before clock reads, counter snapshots and timing allocations. Original
execution allocations and waits remain unchanged.

## Structured log schema

Target: `ferrule_performance`, level DEBUG.
`event="mixed_rank_performance"`, `schema_version=1`.
Exactly one summary per completed layer and one per owner invocation. A failed
active layer is summarized on the invocation's failure path. Identity fields:

- `scope`: `layer` or `invocation`; `layer`: global layer, absent on invocation
- `owner`, `device`, `transaction`, `rows`, `sequences`, `prefill_rows`, `decode_rows`
- `status`: `ok`, `error`, `panic`; `quarantined`; `diagnostic_observer_enabled`
- `host_wall_us`: measured scope wall interval, not TTFT or publication latency

Phase fields are calls and **exclusive host-wall microseconds**:

- `attention_projection_calls`, `attention_projection_host_us`
- `gqa_calls`, `gqa_host_us` (core and its output gate)
- `gdn_calls`, `gdn_host_us`
- `router_calls`, `router_host_us` (projection/router work excluding nested read)
- `route_host_read_calls`, `route_host_read_us` (IDs/weights download + conversion)
- `route_agreement_calls`, `route_agreement_host_us`
- `shared_ffn_calls`, `shared_ffn_host_us` (shared FFN and scalar gate)
- `dense_ffn_calls`, `dense_ffn_host_us` (dense-only FFN)
- `routed_ffn_calls`, `routed_ffn_host_us`
- `collective_calls`, `collective_host_us` (D2H, packing, exchange, H2D and original completion)
- `head_calls`, `head_host_us` (output pipeline/projection excluding nested collective)
- `retirement_host_us`, `phase_other_us`, `phase_failures`

Phase calls count timed entries, not kernels or unique tensors. Re-entering the
same category increments calls; nested category durations do not double-count.
Other includes norms/residuals, admission/completion and unclassified host work.
Microsecond truncation can make summed fields differ slightly from wall time.
A failed layer summary can include its subsequent original cleanup.

Wait summaries reuse `ProofTiming`:

- `consumer_event_{calls,failures,host_us}`: existing event record + event wait
- `compute_sync_{calls,failures,host_us}`: existing semantic compute wait/cleanup
- `upload_sync_{calls,failures,host_us}`: actual upload waits, including consumer proof
- `boundary_drain_{calls,failures,host_us}`: existing transfer drain
- `upload_wait_actual_skips`: actually omitted redundant tail upload waits

**These wait times overlap the phase times and must not be added to them.** They
include GPU execution pending at the host wait, not just synchronization API
cost. These are the explicit model proof boundaries, not all implicit CUDA or
library waits; D2H/transport waits also appear within route/collective/head wall.
No claim of kernel attribution, GPU duration, or CPU-only duration is made.

Counter deltas:

- `allocation_attempts`, `allocation_requested_bytes`
- `facade_kernel_launches`, `facade_stream_syncs`, `facade_stream_sync_failures`
- `h2d_copies`, `h2d_bytes`, `d2h_copies`, `d2h_bytes`
- `artifact_uploads`, `artifact_upload_bytes`, `counter_reset_detected`
- `route_metadata_upload_calls`, `route_metadata_upload_bytes`: successful routed
  gather-ID and combine row-ID/F32-weight uploads, not model-weight uploads
- `source_retirement_batches`, `source_retired_logical_bytes`: encoded numeric
  payload bytes removed from primary layer/head bindings, counted once per
  binding; recurrent descriptor aliases are not counted again

Facade counters do NOT see every native kernel or raw transport DMA; in
particular direct BF16 collective `to_host_vec` need not increment facade D2H.
Do not interpret them as exhaustive CUDA API counts. Reset deltas are missing,
not zero. Allocation bytes are requests, not actual allocator pool growth.

Source retirement is metadata-level logical payload accounting. It excludes
retained embedding/norm/vector payloads, external aliases and allocator behavior;
it is **not RSS reduction, device free bytes, OS free or a reclamation ACK**.

Layer and invocation summaries overlap: sum layers OR use invocation totals,
never both. Invocation Other includes earlier layer-summary logging overhead;
a summary excludes its own logger emission. Subscriber formatting/I/O can
perturb profiling and must be reported separately.

## CPU validation / clock calibration

`cargo test -p ferrule-model --no-default-features --test mixed_rank_timing`
checks disabled clock/guard work, nested exclusive accumulation, counts/failures,
unwind restoration, reset deltas, production gate/schema wiring and that profiling
does not disable the optimized upload-proof path. It does not initialize CUDA.

Manual CPU-only calibration (prepared, not automatically run):

```sh
CUDA_VISIBLE_DEVICES='' cargo test -p ferrule-model --no-default-features \
  --test mixed_rank_timing calibrate_host_clock_and_phase_guard_overhead \
  -- --ignored --exact --nocapture --test-threads=1
```

Use the same optimization profile/toolchain, CPU affinity and otherwise-idle
host as the later profile run. Repeat and report median plus spread of the
100,000-iteration disabled/Instant/guard loops. Each phase normally takes two
clock samples and each measured existing proof takes two; scope snapshots add
clock samples. Estimate instrumentation cost with observed call counts, but do
not subtract it to fabricate model throughput or GPU kernel duration. Separately
calibrate the selected tracing sink with a fixed number of synthetic schema
records; this CPU clock test intentionally excludes logger I/O. Then request a
separate bounded GPU profile run, not a benchmark or quality-oracle rewrite.
