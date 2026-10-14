# Pair-SwiGLU routed fusion — numeric/E2E validation result

2026-10-11. **Ready for a fresh release-builder handoff.** This is an internal
numeric/lifecycle qualification, not a release binary, serving acceptance or
performance result. No production, runtime or independent oracle was modified.
Original backend/model freeze manifests still verify.

## Evidence and final receipt

Repository-relative roots:

- Small gates and E2E: `papers/arxiv/experiments/tp2_resident/runs/pair-swiglu-fused-20261011-03`.
- Final HF loader after clippy-only `chunks_exact` -> `as_chunks` cleanup:
  `papers/arxiv/experiments/tp2_resident/runs/pair-swiglu-fused-20261011-04`.
- Builder/release receipt:
  `target/validation/pair-swiglu-fused-validation-20261011-01/ready-release-builder.json`.
- Receipt SHA256: `fe77dcb66142f370d4de7445265c4cb8db50d52e863eb88eedb8d80bf6014c43`.
- GPU release: same target directory's `gpu-release.json` and `final-idle.json`.

The final receipt pins the successful selectors, logs, raw-word report,
independent comparisons, source inventory, archived workflow/loader, test
binaries and GPU custody. E2E/current source inventories differ **only** in the
isolated fixture-loader chunk iterator cleanup. Its prior source is archived;
the final loader was rebuilt and independently GPU revalidated. Production and
E2E executor/test sources are unchanged, and backend/direct-tail/model library
test binaries retain identical hashes. This is explicit source-delta evidence,
not a claim that every test ran from one identical whole-repository snapshot.

## Coverage actually executed

Every GPU invocation selected exactly **one** ignored test. All successful
receipts use authenticated maintained supervisor/subreaper custody, selected
UUID locks and matching `CUDA_VISIBLE_DEVICES`/`NVIDIA_VISIBLE_DEVICES`.
CUDA 12.4.99 (`tao`), sm86, pinned tool inventory, and
`CUBLAS_WORKSPACE_CONFIG=:0:0`; direct-tail builds enable
`FERRULE_CUDA_TEST_ORACLE=1`. Build/link bounds were 600 seconds per command.

| Gate | Result |
| --- | --- |
| Backend pair rank-pair, BF16 edges, validation/poison | 3 exact selectors passed |
| Independent frozen literal direct-tail words, M1/M2/M5 | Passed; real `numeric_bf16_gemv` target, not zero tests |
| Independent HF fixtures: 8 M1 + 2 routed M5, cold FP8 and resident BF16 | 20 entries, **9,216 BF16 words, zero mismatches** |
| Model noobs/native-observer exactness, fused dispatch/launch/allocation counts | Passed; both rank-local lanes, M1/M2/M5, native and cache cold/hot, retired sources |
| Actual old packing/nonfinite conversion and removed launch counts | Passed; signaling-NaN quieting remains outside direct-packing word equivalence |
| Unknown after actual fused tail, and failure/Unknown after actual down write | Passed; no replay, leases settle on ordinary failure; Unknown retains full arena backing even on owner drop |
| Packed down-output middle slice + neighboring canaries | Passed; M1/M5, native/cached cold/hot, input unchanged |
| CACHEON/no-observer full40, Hello/Capital prefill + one decode | 4 calls, **1,986,560 logits exactly equal**, greedy IDs `11,353,11751,13` |
| CACHEON/no-observer six existing lifecycle recipes | 33 calls, **12,167,680 logits exactly equal**, every independent greedy token matched |

All E2E runs used the existing composition executor and recipe loop, with
`Some(3 << 30)` decoded cache and `trace=None`. Execution contracts record
`numeric_observer_installed=false`, `committed_tensor_export=false`, and zero
trace reserves. The existing comparison reads each call's complete immutable
oracle logits, rejects nonfinite values, and checks original inputs, tokens,
publication/frontier and existing reset/fork/cancel behavior. No new internal
state comparator or replacement execution loop was introduced.

| Existing recipe | Calls | Full-vocabulary values | Exact |
| --- | ---: | ---: | --- |
| `greedy4` | 10 | 3,476,480 | Yes |
| `fork2` | 6 | 1,489,920 | Yes |
| `canary2` | 5 | 2,234,880 | Yes |
| `recreate1` | 4 | 993,280 | Yes |
| `cold-owner1-01` | 4 | 1,986,560 | Yes |
| `cold-owner1-02` | 4 | 1,986,560 | Yes |

Combined full40 + recipes: **37 calls / 14,154,240 logits**, zero failures,
zero nonfinite logits, zero maximum absolute error. Existing tolerances were
not changed; actual results are exact, not merely tolerance passes.

The real-HF loader validates the immutable parent manifest
`807e5b5643ecce603088052e39fbee36ce15dca4a3ba319b51be120e2922d40c`,
child sums, every raw file hash/dtype/extent, original BF16 scales and rank/M.
It uses `gateupinput.bf16` (M5 token order `[1,2,4,3,0]`), the original full
`2N=512`, local `N=256,K=2048` plan, and compares both public fused entries to
`hf.activation.bf16` word-for-word. All 20 actual raw BF16 outputs are retained.

## Resources, custody, and preserved attempts

- Maximum **sampled** device memory: **21.1044921875 GiB per GPU**, below the
  22 GiB cap. This is not an exact/continuous peak measurement.
- Decoded-cache arena: **3,221,225,472 bytes per owner**. Ready/shutdown snapshots
  in the existing logs confirm actual arena backing, not just a plan setting.
- All successful child trees have `proven_drained`, kernel `ECHILD`, no cleanup
  errors and all-eight-device idle checks. Final nonblocking UUID locks were
  reacquired and released; recorded owned birth identities are absent.
- GPU ownership is released; all eight devices have zero memory/use and no
  compute apps. No foreign process was signalled; no extra GPU sync was added.
- Run `02`'s HF numeric test passed with zero word differences, but the initial
  0.5-second observer interval saw only a zero-byte pre-CUDA sample. Its blocked
  receipt is preserved. Short gates were resealed at 0.05-second host sampling;
  no resource threshold or arithmetic rule was weakened.
- First `fork2` and `cold-owner1-01` runs passed all numeric/lifecycle assertions
  but hit the maintained observer's root-exited endpoint race. Both blocked
  receipts remain under `*-endpoint-race-01`; one bounded unchanged confirmation
  per recipe passed. The observer was not modified and no failed receipt was
  relabeled successful.
- Initial loader build import error and strict-clippy chunk-iterator failure
  were fixed; final CUDA loader clippy with `-D warnings`, four CPU paper-policy
  tests, scoped rustfmt and whitespace checks passed.

## Binary hashes and release boundary

| Artifact | SHA256 |
| --- | --- |
| Backend library tests | `71aefda9fc154928712f7cebfbb1f374a2e6b9b57730c1f3b5cc178bae9cade4` |
| Direct-tail integration tests | `5ff0e195fd865a8816d37cc4e498392da800b31df587041d75baa76e2b86de38` |
| Model library tests | `8726aa9785e0615b9bac88423ba48ec2f8a5c561772c04a3fc15da7d88bb9bcf` |
| Final real-HF loader tests | `90c175c2e78ca356fd2f08f706b4e7e6182c857fdad2462364b1f56776fe8a0a` |
| Composition E2E tests | `5ee4f22f67251babc99618b0d3e2500325716e79b2617c6c33586e2af9dcbbcf` |
| Preserved before-fixed release artifact | `7e1c23a88143cb6455e3c6e10bfe43a567fc97b5fd45745800eae3a1e39a0e73` |

The historical `7e1...` artifact and its old 5.7 tok/s result were preserved,
not overwritten or reused as new-path performance evidence. The release builder
must build a fresh production CUDA artifact against the final source pins;
these quality test binaries are not serving/release artifacts.

**Not run/claimed:** cache-off native full40, 2178-boundary diagnostic fallback,
legacy-observer GPU comparison, internal committed tensor/state equality,
performance/timing benefit, or public serving acceptance. Native-observer
small tests exercise the old split tail; full40/recipe no-observer results are
fused-path evidence, not diagnostic trace evidence.

Source/test/report receipt: `pair_swiglu_fused_validation.sha256` in this folder.
