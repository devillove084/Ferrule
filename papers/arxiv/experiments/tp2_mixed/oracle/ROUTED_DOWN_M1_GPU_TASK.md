# Routed-down M1 independent HF capture task

Status: **CPU preparation complete; CUDA capture required and not yet run.**
This is paper-local oracle work only. It must not consume candidate output, import
Ferrule production kernels, modify `full40-03`, or alter any oracle threshold.

## Immutable inputs and pins

- Old reference: `runs/full40-20261009-03`
- Old reference manifest SHA256:
  `fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d`
- CPU preparation run: `runs/routed-down-m1-cpu-01`
- CPU extraction bundle SHA256:
  `6fe6b0c5b8d886586fef422be52f61151fc72345b4077eae33987e8fa325c65f`
- Checkpoint full shard-set SHA256:
  `b51014a9d40c300ea77f12bf8cada444f79c8300277611f6e5f27a62fcb95090`
- Config SHA256: `236c373b0682a4f40c2fa49ef8e5f722a87a8dbfe265286b6d5260b23d7cb229`
- Index SHA256: `b009f7a49c442fce959a2ff6fee41c69248879b90d7ff5636ec6d2068ac4876d`
- Pinned dependencies: Torch `2.13.0+cu129`, Transformers `5.5.3`.
- HF source pins and interception audit are in each old reference manifest.

The verified actual geometry is:

```text
hidden/output N       = 2048
expert global K       = 512
expert local K / rank = 256
TP                     = 2
scale grid / rank      = [16, 2] BF16 words
scale grid / global    = [16, 4] BF16 words
FP8 weight             = [2048, 512] E4M3FN bytes, column-sharded
```

The preparation run extracted original checkpoint bytes only. Its four
subdirectories contain raw rank weight/scale slices and observed route/boundary
metadata, but intentionally do not contain `input.bf16`, `hf.partial.bf16`,
`manifest.json`, or `SHA256SUMS` for the frozen Rust fixture. Those missing files
must come from this independent capture.

## Minimal bounded CUDA job

Use one fresh authenticated paper-oracle child on one granted CUDA device, with
logical TP2 rank branches on that device. This is a numerical reference capture,
not physical TP2 or performance evidence.

1. Verify all GPUs idle, acquire the existing paper owner lock, and use the
   existing bounded paper launcher/custody path. Do not signal foreign PIDs.
2. Set `TF32=off`, BF16 reduced-precision reduction off,
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` (the existing oracle launcher setting), deterministic algorithms, and offline pinned
   HF dependencies. The requested small child timeout is 120 seconds, outer bound 180000 ms,
   estimated device budget 1024 MiB (not a measured peak), and capture cap 16 MiB.
3. Reuse `tp2_oracle` lazy original operand decoding, config-derived logical
   rank partitioning, native HF norm/router/Experts functions, and the frozen
   CPU payload bundle. Do not construct or execute the full40 model and do not
   use model/runtime candidate kernels. All needed original bytes are already
   frozen (102,780,928 bytes total; no new large checkpoint read is required).
4. For the two archived decode calls (Hello input `[11]`, frontier `1 -> 2`;
   Capital input `[11751]`, frontier `5 -> 6`), independently replay only the
   post-attention MoE subpath at layers 0 and 3:
   - Take the old CUDA embedding (layer 0) or previous-layer hidden (layer 3),
     and the old CUDA reduced attention output from `observed.safetensors`.
   - Perform the native BF16 residual add on CUDA and call pinned HF
     `Qwen3_5MoeRMSNorm` with original post-attention norm weights.
   - Call pinned HF router and existing raw-ties-only observation hook; require
     raw logits, full F32 probabilities, selected IDs and route-weight words to
     match both archived ranks exactly before accepting the reconstructed input.
   - Invoke unmodified HF `Qwen3_5MoeExperts.forward` with config-derived local
     geometry and all eight actual experts, once per logical rank. Bind operands
     through the existing `ExpertBank`/decode policy, not a rewritten expert loop.
   - Observe the selected expert's down `F.linear` input/output transparently;
     do not alter that function's arithmetic, expert order or route weighting.
     Audit source/call interception and restore it afterward.
   - Require the native weighted aggregate to match the old CUDA
     `rank.R.layers.L.routed_partial` exactly for each of the eight rank calls.
     This establishes replay consistency; the aggregate is never divided or
     substituted to obtain individual expected values.

   Bound: four case/layer replays, eight native rank expert calls, 64 actual
   expert down projections; select four fixture experts / eight captured rows.
   No attention, GDN state update, new token, full40 forward or CPU numerical
   expected is required. These are archived-CUDA-boundary replays, not a claim
   that a new full-model forward was executed.

5. At layers 0 and 3, require actual rank-0/rank-1 selected router IDs to equal the
   old reference route IDs before writing any fixture. Capture the deterministic
   lowest selected expert from each old route set:

   | Case | Layer | Selected IDs from old reference | Captured expert |
   | --- | ---: | --- | ---: |
   | hello decode | 0 | `[97,210,102,166,142,205,113,21]` | `21` |
   | hello decode | 3 | `[211,199,37,58,84,75,214,113]` | `37` |
   | capital decode | 0 | `[254,208,191,181,81,120,72,182]` | `72` |
   | capital decode | 3 | `[56,224,163,10,115,243,209,255]` | `10` |

6. For each of the four case/layer selections and both logical ranks, capture:
   - the actual local SwiGLU output immediately before that expert's down
     projection, as `rank{0,1}.input.bf16` with exactly 256 BF16 words;
   - the actual local HF-eager down projection before route weighting or
     cross-rank reduction, as `rank{0,1}.hf.partial.bf16` with exactly 2048
     BF16 words.

   The capture must occur on the independent HF expert path, not by deriving
   either value from `routed_partial`, candidate output, or a post-hoc tolerance
   calculation. Preserve BF16 RNE words exactly. The selected expert's actual
   local FP8 down weight and BF16 scale slice must be copied from the original
   checkpoint bytes already pinned by the CPU preparation.

7. Each fixture directory must contain exactly the contract files:

   ```text
   manifest.json
   SHA256SUMS
   rank0.input.bf16
   rank1.input.bf16
   rank0.weight.fp8
   rank1.weight.fp8
   rank0.scale.bf16
   rank1.scale.bf16
   rank0.hf.partial.bf16
   rank1.hf.partial.bf16
   ```

   `manifest.json` must use schema
   `ferrule.mixed-routed-down.tp2.v1`, role `RoutedExpertDown`, `tp=2`,
   `rows=1`, `oracle=hf-eager-bf16-rank-partial`, `n=2048`, `local_k=256`,
   the actual layer/expert, old reference manifest SHA as `checkpoint_sha256`, source payload
   names/hashes, global 128-block coordinates, route IDs/slot, call/frontier,
   and capture source/interception pins. `SHA256SUMS` must include exactly
   `manifest.json` and the eight consumed files. Its external SHA256 is the
   value supplied to `FERRULE_MIXED_DOWN_SHA256`.

8. Before declaring the fixture complete, CPU-check `sha256sum --check
   SHA256SUMS`, exact file sizes, finite BF16 words, FP8 layout validation,
   original payload hashes, route IDs and rank input/partial lengths. Then run
   the existing ignored Rust `frozen_tp2_down_fixture_preflight` only. Do not
   run the two-GPU operator test from this task; that requires a separate owner
   grant and is not paper oracle generation.

## Explicit non-claims

- The reference is HF eager logical TP2 on one CUDA device, not physical TP2.
- W8A16 SIMT and GemmEx output may differ word-for-word; no expected value is
  changed to make the candidate pass. The real HF words are intentionally used
  to expose that mismatch.
- No route weighting, cross-rank reduction, candidate output, tolerance, or
  production kernel is used to synthesize a fixture value.
- If actual route IDs or any captured boundary differs, preserve the failed fresh
  directory and do not create a partial `SHA256SUMS` fixture. Report the exact
  mismatch and keep `model_fixture_ready=false`.
- The current CPU run is **not** a Rust fixture and is not a CUDA/pass claim.

## Launch readiness

No GPU command has been run or issued. The current `launch.py` accepts full-model
exports only, so do not pass an invented small-task flag or bypass authentication
with a direct CUDA script. On authorization, add a narrowly bounded paper-local
replay worker to the existing authenticated launcher; keep this task's frozen
inputs, exact controls and resource bounds. Use a fresh sibling output directory;
never complete or overwrite `routed-down-m1-cpu-01` in place. The runtime's eventual
operator test uses its own `:0:0` setting; that difference must not be hidden or
used to rewrite the HF expected words.
