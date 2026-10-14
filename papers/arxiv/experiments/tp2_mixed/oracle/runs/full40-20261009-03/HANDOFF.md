# Verified independent logical-TP2 reference

**Reference directory (repository-relative):**
`papers/arxiv/experiments/tp2_mixed/oracle/runs/full40-20261009-03`

**Manifest SHA256:**
`fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d`

`manifest.json`: `status=complete`, `full_reference_complete=true`,
`strict_ferrule_full40=false`, `runtime_gate.status=not_run`.

## What was verified

- Pinned Torch 2.13.0+cu129, Transformers 5.5.3, and installed HF source pins;
  no production kernels/imports. TF32 and BF16 reduced-precision reduction off.
- Authenticated worker and numerical-child ancestry under the existing paper
  subreaper, with a 1700-second child and 1800000-ms terminal bound.
- **89 small CUDA checks, all exact**: TP1/native HF prefill and cached outputs
  and caches; logical TP2 GQA/GDN capture geometry and reductions; frozen
  independent FP8 row partial/reduction, local shared+routed MoE, ascending
  expert accumulation, raw-logit ties, native routing weights, ragged/padded
  vocabulary gather, and pinned CUDA F32 GDN state fixtures.
- CPU/CUDA fixture-state differences were measured separately, not assumed
  equal: largest observed difference `4.656612873077393e-10`. Full-model SSM
  values were computed on CUDA and captured as F32.
- **40 layers, two prompts, two stages, 2,338 reference tensors per case**.
  All 4,676 tensor hashes, shapes, dtypes, finite checks and statistics reopened
  and verified; rank reductions, replica routes, raw ties, cache reconstruction
  and full-vocabulary gather checked. See `artifact-verification.json`.
- **14 original shards, 64,196 header tensor entries, 37,463,724,680 bytes**,
  including 31,745 original FP8 matrices with BF16 128x128 scale geometry.
  Whole-shard hashes/sizes, index/config/tokenizer identities re-read after
  export. Header names/dtypes/offsets are in `checkpoint-metadata.json`.
  No floating-point-to-FP8 re-quantization was used.
- 256 MiB decoded weight cache; peak CUDA allocation 4,069,939,712 bytes;
  peak host RSS 5,011,218,432 bytes. No 35 GB all-model GPU residency or 128 GiB
  host allocation was needed. These are resource-bound observations, not a
  performance/residency acceptance claim. Supervisor wall time was 220.875 s.

## Inputs and continuation

| Case | Prefill token IDs | Recorded decode input | Decode next ID |
|---|---|---:|---:|
| Hello | `[9419]` | `11` | `353` |
| Capital | `[760,6511,314,9338,369]` | `11751` | `13` |

The candidate **must replay the recorded decode input**, even if its own
prefill prediction differs. Both full vocabulary tensors (248,320 columns)
and the ordered rank-local shards are available in each safetensors file.

## Unsharded baseline: descriptive drift, not TP2 acceptance

The original unsharded reference was read-only:
`target/validation/numeric-bf16-gate-20261004-v2/native-reference`, manifest
SHA256 `8772456576dc0568241b53a47db01751f33bbd0e90ee5e21680ead0cb13b727b`.
All four comparisons used matching inputs and agreed on full-vocab argmax.

| Case/stage | Logit max absolute drift | RMSE | Elements outside unchanged tolerance |
|---|---:|---:|---:|
| Hello prefill | 0.25 | 0.0455972 | 104083 / 248320 |
| Hello decode | 0.25 | 0.0552579 | 154521 / 248320 |
| Capital prefill | 0.6875 | 0.0709795 | 781127 / 1241600 |
| Capital decode | 0.4140625 | 0.0837440 | 196126 / 248320 |

Detailed layer/route/state drift is retained in `hello.json` and `capital.json`.
These are real tolerance failures against the **different unsharded contract**,
not silently accepted equality. No individual full-model drift cause has been
isolated here; the TP2 reference is the explicit sharded HF protocol, not a
claim that all drift is harmless. No tolerances were widened.

## Runtime gate consumption

`expected-comparison.json` contains the exact protocol, checkpoint identity,
recorded calls, and the **2,178 required candidate tensors per case**. The only
omissions from the reference inventory are pre-canonicalization HF
`native_expert_ids` (diagnostics); canonical selected IDs remain mandatory.

The runtime owner must independently export a candidate manifest with schema
`ferrule.paper.tp2-mixed-candidate.v1` and provide its SHA256. Do not generate
candidate arrays by copying reference values. The candidate's execution,
physical topology and custody evidence are separate obligations.

From the repository root, substitute the independently produced candidate
path and hash:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /opt/conda310/bin/python -B \
  papers/arxiv/experiments/tp2_mixed/oracle/compare.py \
  --reference papers/arxiv/experiments/tp2_mixed/oracle/runs/full40-20261009-03 \
  --reference-sha256 fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d \
  --actual CANDIDATE_DIR --actual-sha256 CANDIDATE_MANIFEST_SHA256 \
  --output papers/arxiv/experiments/tp2_mixed/oracle/runs/full40-20261009-03/runtime-comparison.json
```

Floating comparisons remain `abs(actual-expected) <= 0.00025 +
0.012*abs(expected)`; inputs/IDs are exact. F32 states report exactness too.
Missing boundaries, mismatched prompts/frontiers, nonfinite values, reordered
vocabulary shards and malformed artifacts fail closed.

**Still unresolved for production admission:** full-width CUDA GEMM,
softmax/norm/activation implementation drift; production GDN chunk/recurrence,
exp/softplus/L2 and exact state update matching; physical TP2 transport and
production expert residency. vLLM fused MoE is explicitly a different order,
not a bit-exact target. The small CUDA checks do not cover every full-width
production math site. Every failed or uncovered site blocks the runtime claim.

## Failures, drain and scope

- Attempt 01: rejected before checkpoint reads because the child compared the
  subreaper's pre-`setsid` journal session with the live session. Fixed to compare
  stable PID/birth/parent identity while retaining live ancestry/session checks.
- Attempt 02: generated both cases but stayed **incomplete** after the raw
  `marshal.dumps(code)` fingerprint changed. CPU reproduction proved that merely
  retaining references to unchanged constants changes marshal bytes. Canonical
  bytecode/constants plus source hashes are now enforced; raw marshal hashes
  remain diagnostic. Attempt 02 was never promoted or substituted for 03.
- Both failures and all their receipts/artifacts remain unchanged in sibling
  directories. No original unsharded references were modified.
- `all-eight-release.json` records eight devices at zero memory/utilization,
  no compute apps, proven-drained descendant custody, and successful nonblocking
  reacquisition/release of the shared owner lock. **GPU use has ended.**

One physical RTX 3090 simulated two logical ranks. No physical TP2,
performance, vLLM bit-exactness, Ferrule full40 PASS, or factory enablement is
claimed. No system/runtime files or Git index entries were changed in this turn.
