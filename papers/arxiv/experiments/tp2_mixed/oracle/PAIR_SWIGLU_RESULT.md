# Pair-SwiGLU independent HF CUDA fixtures — ready

## Release and custody

Fresh root: `runs/pair-swiglu-m1-m5-cuda-01` (relative to this document).
Ten complete rank-local fixtures: eight M1, two genuine routed M5.
Root `status=complete`, `fixture_ready=true`.

- Manifest SHA256: `807e5b5643ecce603088052e39fbee36ce15dca4a3ba319b51be120e2922d40c`.
- CPU validation/release receipt: `runs/pair-swiglu-m1-m5-cuda-01.validation.json`.
- Receipt SHA256: `dae2be312a951e7cb3d4a22afb406e96fe97ef3ce8fe73816ce64b01662e8e91`.
- Replay preparation: `runs/pair-swiglu-m1-m5-cpu-02`.
- Preparation `SHA256SUMS.cpu-extract` SHA256:
  `4b2b1c956f24f5bd2c8af2e21f36689f7d07f27fbd71a383bec8b55b32a6e440`.

One authenticated task used physical GPU0,
`GPU-b980f22b-58a3-fc88-21c1-ae86053d8535`, with two logical rank branches.
The selected CUDA UUID, both visibility variables, sealed grant and shared lock
agreed. Existing authenticated launcher/subreaper was reused, not replaced.
Child bound 120 seconds, outer 180 seconds; neither timed out.
Numerical work: 5.400 seconds; supervisor: 7.754 seconds; peak reserved 120 MiB.
Capture: 230400 bytes. No full-model replay, candidate import or threshold change.

**GPU released:** supervisor return code 0, custody `proven_drained`, no cleanup
errors. All owned child/worker/helper/coordinator birth identities were absent.
The shared UUID lock was reacquired nonblocking for final idle proof and released.
Eight devices were idle before, after, and at final release: zero memory/use,
no compute apps. No foreign process was signalled.

## Ready fixture directories and pins

Paths below are relative to the release root. The pin is SHA256 of the child
`SHA256SUMS`, **not** the root manifest hash or a raw tensor hash.

| Child directory | M | SHA256 of SHA256SUMS |
| --- | ---: | --- |
| `hello-decode0-layer00-expert021-rank0` | 1 | `7b8bfbcce6312aa58fb6fee8c13cd5f489b85923a4825712eae3eb9b327ec5a0` |
| `hello-decode0-layer00-expert021-rank1` | 1 | `da03c55e56b74d5665eeadb7c8ed7e17425c4ba182d5225f06656b9181d81238` |
| `hello-decode0-layer03-expert037-rank0` | 1 | `774b26d3acdfd6b7531bdd9dad761ce97b620f34c98369e7cf9a1a2320e4c3b1` |
| `hello-decode0-layer03-expert037-rank1` | 1 | `fcbcfef9797f218f972cd3b99e120d31f02f094e12a8af258449a01e1f6bc4c4` |
| `capital-decode0-layer00-expert072-rank0` | 1 | `a87b5fd5ad9de1d8f8c282ba895e265c5a23f58eb5b3fcb3d43acd3a6183ae9e` |
| `capital-decode0-layer00-expert072-rank1` | 1 | `e34a427313156602a0b0fc3917b67112e686fb2496a336276d2993c11aa65d64` |
| `capital-decode0-layer03-expert010-rank0` | 1 | `11feb71b1ed76004c3996ba98c1466a0bf25266e7d8787315fe592eb15845522` |
| `capital-decode0-layer03-expert010-rank1` | 1 | `77e390f471b96cf0556330c347db5af76059dae8a00e02c44a3014e8567a00d3` |
| `capital-prefill-layer09-expert095-rank0` | 5 | `bb5038c0898533ff85a292315e98fb0a560cf5c1cb66f05a96c601f64639d534` |
| `capital-prefill-layer09-expert095-rank1` | 5 | `e121e43b9ff026c82c3a68323126ce06fff5185b899e1e041512507ac307a17d` |

## Oracle provenance and exact controls

Reused unmodified pinned native HF norm/router/Experts.forward and the original
FP8 decode/TP partition helpers. Torch 2.13.0+cu129, Transformers 5.5.3,
CUDA 12.9; TF32 and BF16 reduced-precision reductions disabled;
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. No CPU normalization or candidate output
supplied a reference. Observation of F.linear arguments was transparent and
restored; source/interception audit was unchanged afterward.

Actual calls: five norm, ten router, ten native Experts.forward, 112 packed
gate/up linears and 112 down linears. Four M1 cases each use eight active
experts per rank; Capital prefill layer9 uses 24 active experts per rank.
Expert95 receives all five actual tokens, so its native expert batch is M5.

All **66 controls were byte-exact**, with no tolerance:

- 40 old raw logits / F32 probabilities / IDs / route weights controls;
- 10 old weighted routed aggregate controls;
- 8 M1 activations and 8 M1 down outputs against the prior independent CUDA run.

The original source reference manifest is
`fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d`.
M1 payloads reuse the immutable `routed-down-m1-cpu-01` bundle. M5 preparation
reads only 146 original norm/router/expert payloads (76559360 bytes) and touched
headers, checked against archived used-byte hashes; no whole checkpoint shard
scan. Capital CUDA boundary tensors retain the old reference identity.
All old references and both old M1 preparations remain unchanged, including the
previous `pair-swiglu-m1-cpu-01` readiness=false result.

Child manifests include source/oracle/workflow pins, original payload hashes,
shard headers, rank slice coordinates, token/call/route provenance, generation
launch/authentication identity, shapes, byte extents and every raw-file hash.
The four changed launcher/replay/contract/test sources are frozen in the new
release `source/` directory and verified against the launch workflow pins.

## Validator consumption contract

Each child has `manifest.json`, `SHA256SUMS` and seven raw tensors. Tensors are
contiguous row-major; BF16 words are little-endian. N=256, K=2048, full packed
plan width **2N=512**; M is 1 or 5 according to the child manifest.

| File | Dtype / shape | Bytes |
| --- | --- | ---: |
| `gateupinput.bf16` | BF16 `[M,2048]` | `M*4096` |
| `normalized_hidden.bf16` | BF16 `[M,2048]` | `M*4096` |
| `hf.activation.bf16` | BF16 `[M,256]` | `M*512` |
| `gate.weight.fp8`, `up.weight.fp8` each | E4M3FN `[256,2048]` | 524288 |
| `gate.scale.bf16`, `up.scale.bf16` each | BF16 `[2,16]` | 64 |

Scales are the original BF16 128x128 block grid, not converted/scalar scales.
Weights are global N rows `[256*rank,256*(rank+1))`, scales global rows
`[2*rank,2*(rank+1))`, all K columns. Local backend artifacts use contiguous
zero-origin geometry; the global coordinates are provenance.

**Use `gateupinput.bf16`, not normalized_hidden, as the GEMM input.** M5 HF
expert batching is in token order `[1,2,4,3,0]`, while normalized_hidden is
`[0,1,2,3,4]`. The activation uses the actual gate input order. This is the
real unchanged HF gather order, not tiled M1 or invented routed rows.
Both rank manifests record and validate this permutation. M1 order is `[0]`.

Validator admission must verify the complete parent release pin, child sums
pin, all file hashes/dtypes/extents, rank and M, and original scale encoding.
Create the original full-2N plan. Cold API consumes gate/up immutable FP8
artifacts; resident API consumes decoded BF16 gate rows followed by up rows,
with layout `contiguous(M,512,2048)`. Both produce `[M,256]` activation directly;
do not run another SwiGLU on that output.

Compare **every BF16 word** from both candidate entry points to the independent
`hf.activation.bf16`. The old split+SwiGLU and shared-helper comparisons are
additional controls, not substitutes for this independent oracle. Preserve
original expf SiLU, its BF16 rounding, then multiply and BF16 rounding.
If the backend differs, report the actual mismatch; never replace expected,
relax exactness, or silently reorder only one operand.

The oracle retains its historical CUDA/:4096:8 policy; backend validation uses
its own pinned environment in `pair_swiglu_handoff.md`. These fixtures do not
claim that backend arithmetic, GPU rejection paths, performance, physical TP2
transport or model integration has passed.

## Validation performed

- Six existing/extended CPU launcher tests passed, including no-I/O preflight,
  fixed bounds/pins and unauthenticated worker rejection.
- GPU-hidden import/source/dependency and five-case frozen-bundle checks passed.
- One authenticated GPU task passed all 66 exact controls and capture audits.
- CPU postvalidation checked all ten child inventories, original FP8/scale slice
  identity, finite FP8/BF16 values, positive scales, actual token permutations,
  M1 immutable activation identity, workflow/source/generation pins and custody.
- `runs/pair-swiglu-m1-m5-cuda-01/validate-source.py` records that CPU validation;
  its hash is pinned by the validation receipt. No backend validator was run.
