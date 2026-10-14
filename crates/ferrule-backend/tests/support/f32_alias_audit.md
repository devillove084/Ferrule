# F32 prefix alias and recurrent grant audit

Scope: safe `CudaF32Buffer::prefix` aliases reaching owner-facing F32 writers.
No raw-pointer fabrication, unsafe casts, length relaxation, kernel/math/dtype
changes, or new allocation/owner ledger. GPU tests are definitions only for the
next GPU-owner run; this task runs CPU tests and CUDA compile checks only.

## Common range contract

`cuda/buffer_range.rs` factors the existing recurrent half-open byte-range
check into checked F32 byte/address arithmetic. Empty intervals do not overlap.
`CudaOperators::f32_no_alias` rejects read/write and write/write intersections;
read/read aliases remain valid. `f32_inplace_or_disjoint` permits only equal
logical intervals or disjoint intervals, not an overlapping shorter prefix.
Copies use the actual source/destination subranges, not root allocation extent.
All checks run before the affected operation's first write/submission.

## New rejection coverage

| Entry family | Newly rejected aliases |
| --- | --- |
| `rms_norm_f32_into`, legacy `rms_norm*_from_device_into` | Non-exact input/output overlap; broadcast weight/output overlap across row blocks. In particular output `[2,W]` with its own `prefix(W)` as weight is rejected. |
| `copy_f32_range`, `copy_f32_into_slot` | Overlapping non-identical copy subranges. Identical copies become no-ops. |
| `concat_f32_buffers_into` | Either copy overlapping its own source non-identically; the first non-no-op copy clobbering the second source. Both copies preflight before either executes. |
| `residual_add_f32_into` | Output clobbering update during residual copy; partial input/output overlap. |
| `saxpy_into`, `swiglu_f32_into`, `swiglu_f32_clamped_into` | Non-exact elementwise input/output overlap. |
| `weighted_combine_f32_into`, `weighted_scatter_add_bf16_moe_rows`, `scatter_add_f32_rows` | Output/destination overlapping values or broadcast route weights. |
| `paged_plane_scatter_*` | Source values overlapping the writable plane. |
| `artifact_linear_matvec_into`, `artifact_linear_rows_from_device_into{,_with_scratch}` | Direct-read projection input overlapping output. Already-staged FP8/quantized paths retain their staging semantics. |
| `artifact_linear_pair_matvec_into`, `artifact_linear_pair_rows_from_device_into_with_scratch` | Output/output overlap; outputs clobbering an input needed by another projection. Shared-prepack pair path permits input/output overlap after the complete pack. |
| `grouped_output_a_bf16_from_fp8_into`, `grouped_matvec_f32_rows_from_device_into` | Output overlapping context or F32 grouped weights. |
| `dsv4_router_{topk,hash}_sqrt_softplus_rows_from_device_into`, `selected_softmax_topk_from_device_into` and standard/semantic delegates | Output weights overlapping logits or selection bias. |
| `topk_vocab_rows_from_device_into`, `artifact_linear_topk_from_device_into` | Values/logits/input and F32 multi-output overlap. Composite linear+top-k validates before projection. |
| `compressor_recurrent_{seed_prefill,append_projected,boundary_into}`, `compressor_prefill_softmax_from_device_into` | Projected values/scores/APE aliasing writable state; output aliasing consumed state or softmax inputs. |
| `reduce_moe_route_outputs_ranked`, `reduce_expert_group_route_outputs_ranked` | Output overlapping route table reads. |
| `hc_pre_from_device_into`, `hc_post_from_device_into`, `hc_mean_scatter_from_device_into`, `hc_head_from_device_into` | HC output(s) overlapping state/residual/mixing parameters; HC pre write/write overlap. |
| `rope_tail*_from_device`, `split_half_rope*_from_device`, standard and semantic delegates | Writable values overlapping cosine/sine tables; values still update in place. |
| `append_and_attend_paged_f32` | Cache/cache/output overlap; any writer overlapping query/append inputs. Rejected before metadata upload. |
| Sparse/paged/dual-plane attention `*_into`, selected-row variants, `paged_bf16_*gqa*_into` | F32 output overlapping query, F32 planes/values or sink. BF16 wrappers share query/output preflight. |
| Semantic `artifact_bf16_compressor_into`, `artifact_fp8_query_a_kv_into` | Output/output overlap and direct compressor input/output overlap. |
| Semantic `hc_pre_rmsnorm_fp8_into`, `artifact_main_project_norm_into`, `artifact_proposal_head_into` | F32 input/write and write/write overlap across their compound launches, including proposal workspace confidence exposed by a prefix. |
| Semantic `artifact_shared_ffn_into` | Hidden scratch aliasing output in accumulation mode, where gate/up would destroy the original output. |
| Semantic `hybrid_mla_attention_into` | Output overlapping query, context plane, proposal KV or sink. |

Recurrent conv/DeltaNet/query split/gating/offset RMS retain their existing
strict disjoint-writer policy, now using the common checked range helper.
Streaming F32 GQA retains its existing disjoint policy with the same helper
instead of unchecked pointer-end arithmetic. Standard F32 GEMM and numeric
FP8 GEMM already reject aliasing in provider preflight; their typed errors and
stride-specific range checks are unchanged.

## Compatibility retained

- All read/read aliases and ordinary disjoint-buffer calls.
- Exact input/output norm (including head norm), SAXPY and standard SwiGLU
  in-place computation. RMS weight/output exact equality implies a single row
  after the existing exact shape validation and remains safe; multi-row
  broadcast weight/output aliases are rejected.
- Residual output equal to residual, including the all-three-equal case.
- Same-allocation disjoint range copies, exact self-copy no-ops, and safe
  append-in-place concat. `copy_f32_within` keeps its existing strict policy.
- RoPE and single-buffer quantization/zeroing remain in place.
- FP8 input staging before output writes, including the shared-prepack pair.
  `artifact_fp8_projection_rows_from_device_into_with_scratch` already packs
  first. `artifact_mla_output_into` consumes context into separate BF16 latent
  before output-B (stream order or grid barrier), so context/output overlap is
  still permitted. Shared FFN overwrite mode finishes the separate FP8 hidden
  pack before writing output and still permits hidden/output overlap.
- Fresh-result allocators and opaque scratch with no exposed F32 views cannot
  acquire a new prefix alias; no unrelated runtime/provider changes were made.

## Allocator grant accounting

`ferrule_backend::cuda_allocation_grant_bytes` is a read-only CPU-available
query. Live `allocator::allocation_layout` uses exactly this function:

- zero -> zero;
- up to 1 MiB -> next power of two, minimum 256 B;
- above 1 MiB -> checked rounding to 2 MiB pages (including huge requests).

`RecurrentScratchPlan` still computes eight logical lengths and componentwise
cross-layer high-water capacities. Its single peak charge is now twice the
sum of the eight allocator grants, covering one retained old generation plus
one replacement per buffer until the completion fence. It is not added again
to expert cache/numeric reservations. Shared allocator segment slack is not a
per-buffer grant and is not duplicated here.

For channels=10, heads=2, width=4, rows=1 -> rows=2: eight old grants plus eight
new grants are `16 * 256 = 4096 B`, not the old logical upper bound `640 B`.

CPU oracles: `cuda_scratch_metadata.rs` and model segment-inventory unit tests.
GPU definitions: `cuda_f32_alias.rs`, `cuda_f32_prefix.rs` and model
`tests/unit/transformer/recurrent_cuda.rs`; not executed in this task.
