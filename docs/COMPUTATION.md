# DeepSeek-V4 computation contract

<!-- markdownlint-disable MD013 MD060 -->

This document records Ferrule's numerical contract for the DeepSeek-V4 Flash
0731 profile. It distinguishes FP32 computation from BF16 tensor writeback and
identifies the boundaries that can change sparse selection, proposal
acceptance, or final tokens.

It does not claim full-model B300 parity. Current evidence consists of CPU
regressions, CUDA execution on an RTX 3090 (`sm_86`), and compile-only
validation for `sm_103`.

## 1. Reference identity and model shape

The local reference artifact is:

```text
models/DeepSeek-V4-Flash-0731/
reference revision: 7872f01b1d1fe23eabc4c98b48bffcef5a386062
```

The authoritative executable reference is
`models/DeepSeek-V4-Flash-0731/inference/model.py`; checkpoint metadata is in
`models/DeepSeek-V4-Flash-0731/config.json`.

Relevant profile values are:

| Property | Value |
|---|---:|
| Activation dtype | BF16 |
| Hidden size | 4096 |
| Attention heads | 64 |
| Attention head dimension | 512 |
| RoPE tail dimension | 64 |
| Indexer heads | 64 |
| Indexer head dimension | 128 |
| Indexer top-k | 512 |
| Sliding window | 128 |
| Routed experts | 256 |
| Experts selected per token | 6 |
| Compression ratios | alternating 4 and 128 after the first two layers |
| Original YaRN length | 65,536 |
| Advertised maximum positions | 1,048,576 |

The checkpoint and its Python reference define semantics. Faster kernels are
valid only when their observable BF16 values, indices, routes, and transaction
results remain within the declared contract.

## 2. Precision model

### BF16 writeback

The reference uses BF16 activation tensors. PyTorch operations may compute
internally in FP32, but assignment or operator output into a BF16 tensor rounds
the result before the next operation. Removing such a writeback can change a
later quantization scale, near-tie ordering, or token.

Ferrule uses round-to-nearest, ties-to-even BF16 conversion and preserves
infinities while quieting NaNs. Shared CPU helpers are in
`crates/ferrule-model/src/models/common/math.rs`.

Some Rust and CUDA interfaces use `f32` storage for convenience while requiring
values to be logical BF16. In that case every stored value must already equal
its BF16 round trip:

```text
stored_f32 == f32::from(bf16::from_f32_rne(stored_f32))
```

Logical BF16-in-F32 storage is a representation choice, not permission to carry
extra FP32 precision across a BF16 boundary.

### FP32 accumulation

A BF16 output does not imply BF16 accumulation. Reductions, normalization
statistics, softmax, matrix-accumulate fragments, and recurrent compressor
state use FP32 where the reference or kernel contract requires it. The general
rule is:

```text
BF16 input tensor
  -> FP32 arithmetic and accumulation
  -> one BF16 writeback at the reference tensor boundary
```

Rounding every partial product is wrong. Keeping the final result in FP32 when
the reference writes BF16 is also wrong.

## 3. Hyper-Connections and proposal taps

Hyper-Connection mixing and internal state use FP32. Activation outputs that
feed BF16 model tensors are restored to BF16 at their declared boundaries.

The proposal attachment consumes target-layer taps produced by:

```python
main_hiddens.append(h.mean(dim=2))
```

Because `h` is BF16, the mean is accumulated as required and its tensor result
is written to BF16 before stage-zero projection and FP8 quantization. Ferrule's
CUDA mean-scatter path therefore rounds `sum / hc` to BF16 once before the next
operator.

Proposal-only differences usually change draft quality and acceptance rather
than the final exact greedy token, but proposal state still must match the
checkpoint contract to make acceptance and performance evidence meaningful.

## 4. RoPE and YaRN

`apply_rotary_emb` converts pairs to FP32 complex values, multiplies by the
frequency, and copies the result back into its input tensor. For BF16 inputs,
that copy is a BF16 boundary. The same rule applies to inverse RoPE.

Ferrule performs:

```text
BF16 input
  -> FP32 rotary arithmetic
  -> BF16 writeback
```

Compressed-attention layers use the checkpoint's YaRN parameters and compressed
RoPE theta. Absolute token positions, not packed-row or page-local offsets,
select frequencies. The original YaRN length is 65,536 even though metadata
advertises a larger maximum position count.

## 5. KV compressor

The compressor projects KV and gate values and retains decode recurrent state
in FP32. Pooling uses FP32 softmax and accumulation. The official post-pooling
sequence is:

```text
FP32 projection, scores, state, softmax, and pooling
  -> BF16 pooled KV
  -> FP32 RMSNorm arithmetic
  -> BF16 normalized KV
  -> FP32 RoPE arithmetic
  -> BF16 rotated KV
  -> FP8 or FP4 QAT
```

The pooled-KV writeback before RMSNorm is mandatory. Feeding the unrounded FP32
pool directly to RMSNorm changes normalized values and can propagate into
sparse attention.

Decode emits a compressed row when the completed token count is divisible by
the ratio:

- ratio 4: boundary token positions `3, 7, 11, ...`;
- ratio 128: boundary token positions `127, 255, 383, ...`.

Ratio-4 compression combines an overlapping prior window with the current
window. Entries outside the active overlap must use zero KV and negative
infinity score semantics; stale recurrent lanes must never contribute. Paging,
copy-on-write, cache lengths, and compressed row identities are derived from
boundary token positions rather than the physical page currently holding a
source token.

## 6. Indexer

Ratio-4 layers use the learned indexer to select compressed KV rows. Its
activation path is:

```text
BF16 query projection output
  -> FP32 RoPE
  -> BF16 writeback
  -> FP32 normalized Hadamard transform
  -> BF16 writeback
  -> FP4 QAT scale selection and quantization
```

The BF16 writeback after Hadamard must occur before FP4 scale selection. Values
near an E8M0 scale boundary can otherwise select a different scale and produce
different scores.

The official score path is a chain of BF16 tensors around an FP32 head
reduction:

```text
weights projection -> BF16
weight scaling      -> BF16
Q dot compressed KV -> BF16
ReLU × weight       -> BF16
head reduction      -> FP32 accumulation
final score         -> BF16
stable top-k
```

Ferrule initializes unused output indices to `-1`, excludes non-finite
candidates, and uses deterministic index order for equal scores. This matters
for near ties: an FP32 fused score can rank two rows differently even when the
official BF16 chain makes them equal.

Ratio-128 layers do not use the learned indexer. They directly select all
eligible compressed rows according to the fixed compressed-coordinate rule.
Selection is based on completed compression groups and excludes future rows.

## 7. Sparse MLA attention

The selected context combines the sliding-window rows and eligible compressed
rows. Logical token indices and physical selectors remain separate; an invalid
or unfilled slot is `-1` and cannot alias row zero.

The verified CUDA sparse-attention contract is:

```text
query and KV inputs       BF16
score products/reduction  FP32
softmax denominator       FP32
probability tensor        BF16
value accumulation        FP32
attention output          BF16
```

The sink term participates in the denominator according to the checkpoint
implementation. All selected rows must be generation-valid, position-correct,
and stable before launch.

After attention, inverse RoPE performs FP32 math and restores BF16. Grouped
Output-A writes the latent activation at its BF16 boundary. Output-B accumulates
all K=128 groups into one FP32 result and performs one final BF16 writeback.
Rounding or overwriting each partial K group changes the output substantially.
For split launches, a nonzero output offset must address the correct destination
rows.

## 8. MoE and quantized linear operators

Router score math follows checkpoint semantics, including deterministic top-k,
selected-route ordering, normalized weights, and six routed experts per token.
Non-finite or uninitialized routes cannot become valid expert indices.

Quantized operators preserve these boundaries:

- BF16 activation values are rounded before FP8 or FP4 scale selection;
- FP8 E4M3 and FP4 QAT use the checkpoint block and scale formats;
- gate and up projections accumulate in FP32;
- clamp and SwiGLU arithmetic use FP32;
- route weighting and expert reduction accumulate in FP32 where declared;
- public activation and residual outputs return to BF16 once at their tensor
  boundary;
- grouped kernels must honor nonzero source and destination row offsets.

CUTLASS and portable CUDA implementations share this semantic contract. A
kernel schedule, MMA atom, or epilogue may change only when the resulting
boundary values and route decisions remain valid.

## 9. Sequence and sampling limits

The checkpoint advertises 1,048,576 positions, but Ferrule caps exact fixed
compressed selection at 65,536 tokens for the current release model. Requests
that require unsupported fixed-selection semantics fail closed rather than
silently truncating or changing the candidate set.

The current serving path implements deterministic greedy decoding. The official
reference uses full-vocabulary Gumbel-max when `temperature != 0`. Ferrule does
not claim parity for that path and rejects nonzero-temperature DeepSeek-V4
serving instead of substituting top-k or nucleus sampling.

## 10. Regression evidence

Important CPU regressions include:

- `bf16_rounding_uses_ties_to_even_and_quiets_nan`;
- `compressor_rounds_pooled_values_before_rms_norm`;
- `indexer_score_rounds_dot_products_before_near_tie_ranking`;
- `compressed_attention_decode_reference_updates_compressed_cache`;
- fixed-selection, ratio-4 overlap, ratio-128 addressing, and paged-COW tests in
  the DeepSeek-V4 model modules.

Important CUDA regressions include:

- `hadamard_fp4_qat_restores_bf16_before_scale_selection`;
- `indexer_bf16_score_boundary_preserves_stable_near_tie_order`;
- `cuda_hc_mean_scatter_builds_proposal_target_taps_without_host_concat`;
- `cuda_recurrent_state_matches_reference_without_allocations_or_d2h`;
- sparse-attention BF16 reference tests;
- grouped MLA Output-A/Output-B and MoE offset tests;
- paged decode metadata and compressed-selection tests.

Run the platform-independent suite without CUDA:

```bash
FERRULE_NO_CUDA=1 just lint
FERRULE_NO_CUDA=1 just test
```

Run CUDA tests on the detected device and compile the B300-oriented profile:

```bash
just test-cuda-required
just check-cuda-arch sm_103
```

A compile-only `sm_103` result validates toolchain and source compatibility. It
does not execute the kernels or establish token parity on B300.

## 11. Evidence boundary

The review found multiple independent drift sources rather than one proven
single fault:

- missing BF16 writeback before compressor RMSNorm;
- missing RoPE and Hadamard BF16 boundaries in the indexer;
- FP32 indexer ranking where the reference ranks BF16 scores;
- proposal target-tap means retained in FP32;
- MLA partial-accumulation and split-output errors;
- compressed-cache metadata, page-coordinate, overlap, and direct-selection
  errors;
- invalid top-k initialization and packed decode metadata errors.

These issues are sufficient to explain token drift in affected paths and now
have focused regressions. Final proof still requires an end-to-end comparison on
B300 using the exact checkpoint, prompts, runtime configuration, and source
revision. Until that run exists, Ferrule must not claim complete DeepSeek-V4
B300 token parity.
