# Hybrid segment / pipeline state contract

## Frozen model API

- `HybridStateSchema::for_segment(&DecoderModelSpec, Range<usize>) -> Result<Self>`
- `HybridStateSchema::{entries, entry(global_layer), layers, layer_range,
  total_layers, kv_planes, per_sequence_state_bytes}`
- `HybridStateSchema::validate_partition(spec, &[schema])`
- `HybridLayerEntry` public fields: `global_layer`, `local_layer`, `state_slot`,
  `global_kv_layer: Option<usize>`, `local_kv_layer: Option<usize>`.
- `HybridDecoderSegment::prepare(resources, plan, precision, max_positions,
  max_parameter_bytes) -> SegmentResult<Self>`
- `HybridDecoderSegment::{plan, schema, parameters}`
- `validate_execution(batch, &[S])`, `validate_frontier(batch, &[S])`, with
  `S: StandardSequenceState`; no Clone/Send bound.
- CPU: `execute(batch, SegmentInput, &mut [HybridDecoderSequenceState],
  &mut CpuKvView) -> SegmentResult<SegmentOutput>`.
- CPU: `execute_checked` adds `&mut dyn FnMut(ExecutionTransactionId) -> Result<()>`.
- CUDA: `CudaHybridDecoderSegment::prepare` takes the same preparation arguments
  plus a final `HybridCudaDevice` (created inside its owner thread).
- CUDA exposes the same plan/schema/parameters/validation APIs;
  `execute_cuda_bound(batch, DeviceSegmentInput, &mut [CudaHybridSequenceState],
  &mut CudaKvView) -> SegmentResult<DeviceSegmentOutput>`; its `_checked` variant
  adds the cancellation callback. `execute` is the explicit host boundary;
  `execute_checked` adds the same callback and rechecks after download.
  `download_output`, `quiesce`, `needs_quarantine`, `resident_parameter_bytes`
  are available, together with `precision`, `activation_dtype`, `kv_element_type`.
  CUDA states remain non-Clone and non-Send.

Both adapters execute the same segment loop and `execute_standard_layer_inner`.
No per-stage runner, second forward, KV allocator or transaction authority exists.

## Coordinates and endpoints

State storage retains one local slot per layer, including full-attention sentinels.
Only KV planes are compact. GDN lookups take global layer IDs and resolve the
schema slot; weights, diagnostics and resident CUDA deltas retain global IDs.
Packed sequence order is explicitly mapped to caller state indices for GDN.

0.8B PP2: `0..12` / `12..24`, 12 state slots each, nine GDN layers each;
local KV `0..3` in both stages; global KV `0..3` / `3..6`.
Parent logical KV uses six full-attention layers, not 24 decoder layers.
Each stage has 10,321,920 recurrent/conv bytes per sequence (F32).

Only stage 0 owns embedding, only stage 1 owns final norm/head. The logical
OutputHead binding retains canonical embedding storage for tied weights without
owning the embedding operation. Physical residency is charged per owner.
`parameters()` contains selected layers and endpoint dependencies only.

## Runtime PipelineStateProgram adapter

CPU associated state/lifecycle: `HybridDecoderSequenceState` /
`HybridSequenceLifecycle::new(segment.schema().clone())`, with
`PagedKvBackend<TypedCpuPagedKvPool<HybridDecoderSequenceState>>`.
CUDA uses `CudaHybridSequenceState` / `CudaHybridSequenceLifecycle::new(schema,
owner_device)` and the corresponding typed CUDA pool on the SAME device owner.

1. Keep committed source immutable. Lifecycle checkout owns the working state.
2. Validate source and working input schema/core/frontier before executing.
3. Enter prepared KV with working states; execute only working recurrent buffers.
4. `validate_frontier` requires every local GDN frontier == sequence_len while
   core/topology/generation still match the input batch. It must precede core advance.
5. On proven computation completion, journal advances working core. Ready/install/
   publish stays under the existing outer distributed transaction.
6. Publish uses move/swap, never Clone or allocation. Journal retains displaced
   committed state through retirement; lifecycle try_release failure returns the
   exact state, which MUST remain in journal custody for retry.
7. Failure/cancellation can leave partially advanced GDN and written KV. Discard
   via lifecycle release + KV rollback; never reuse dirty working state. Fresh
   checkout is required for retry. Checkout success followed by KV prepare failure
   still requires explicit working-state cleanup.
8. CUDA unknown completion quarantines owner/buffers. Neither join, process exit,
   empty physical transaction count, nor a later fence establishes rollback ACK.

`validate_frontier` is not valid after core advance. Exact-frontier fork only;
no partial retain, speculative replay or prefix reuse. Recurrent-only segments
are explicitly rejected with `HybridSchemaError::RecurrentOnly`, never fake KV.

Public factory admission belongs to runtime: strict dense 0.8B, F32, thread,
PP2/TP1/EP1. Model fixture segments need not impersonate the production parser.
This model API does not admit process hybrid, MoE PP/EP or hybrid TP.

## CUDA range admission API

`HybridCudaMemoryEstimate::for_segment(resources, &options, &plan,
physical_kv_pages) -> Result<HybridCudaMemoryEstimate>` is metadata-only. Use
F32 full-depth options, with actual page size, position and batch limits; do not
pass prefix options or divide a whole-model estimate by PP. The dense segment
entry rejects numeric/MoE profiles; existing whole/prefix/numeric entry points
retain their previous formulas.

`kv_bytes` charges local compact planes times the supplied physical capacity:
CUDA planner supplies 2L for logical L. `per_sequence_state_bytes` is one state;
for a serialized owner journal with S sessions reserve `(S+1)*per_sequence`,
covering working, then displaced. `weight_bytes_upper_bound` uses selected
bindings and canonical endpoint closure, conservatively allowing vector/linear
F32 representations (8 bytes per scalar) plus local RoPE. This is not source
BF16 byte size. `workspace_bytes_upper_bound` uses only selected layer widths
and includes vocabulary width only for an output owner.

The runtime must aggregate weight + state + KV + workspace + allocator margin
for ALL colocated owners before creating/uploading any image. The estimate is
not a memory reservation. `prepare` checks actual resident weight bytes against
the supplied device budget, but is not a substitute for planner admission.
CUDA execute preflights every recurrent owner and local KV plane before mutation;
all layer math uses the existing synchronous fenced operators. Quiesce/release
failure means retained custody, not permission to emit a completion ACK.
