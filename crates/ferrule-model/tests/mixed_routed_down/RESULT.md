# Real routed-down M1 small gate — no promotion

## Verdict

The default-off internal candidate remains available for investigation; the
native baseline is retained unchanged. **The exact frozen gate failed.** No
production profile, factory, full-program entry point, operand, expected value,
budget or threshold was changed. No full-model, lifecycle or HTTP run occurred.

All four CPU fixture preflights passed before any GPU launch. The build used
CUDA 12.4.99, cuBLAS 12.4.5.8, sm_86 and `CUBLAS_WORKSPACE_CONFIG=:0:0`.
Five unique selectors were listed with exactly one match each; all six GPU
invocations executed exactly one test. Inputs are the independent release
`papers/arxiv/experiments/tp2_mixed/oracle/runs/routed-down-m1-cuda-01`, manifest
SHA256 `4f3081e16cc496121817bd49e4732cb5b49a030ca9853f616751c17d23ba1704`.

## Exact BF16 results

Each rank has N=2048, K=256, M=1. Coordinates below are zero-based.

| Case | Native vs HF rank0 / rank1 mismatches | W8 vs HF rank0 / rank1 mismatches | W8 maximum absolute error |
| --- | --- | --- | --- |
| hello-decode0-layer00-expert021 | 0 / 0 | 0 / 0 | 0 |
| hello-decode0-layer03-expert037 | 0 / 0 | 0 / 0 | 0 |
| capital-decode0-layer00-expert072 | 0 / 0 | 0 / 1 | 0.0000152587890625 |
| capital-decode0-layer03-expert010 | 0 / 0 | 0 / 1 | 0.0000000009313225746154785 |

- Capital layer00/expert072, rank1, row0, column243:
  W8 `0xbb1e` = -0.002410888671875;
  native/HF `0xbb1d` = -0.0023956298828125.
- Capital layer03/expert010, rank1, row0, column1810:
  W8 `0xb47e` = -2.3655593395233154e-7;
  native/HF `0xb47f` = -2.3748725652694702e-7.

Native matches **16384/16384** words; W8 matches **16382/16384**. Both W8
mismatches fail acceptance, regardless of magnitude. The first failure occurred
in the third case; only the authorized remaining small cases were collected.
Input immutability, F32 output canaries, and prelaunch stride-error atomicity
checks completed for both ranks in every case.

Both synthetic model-path tests passed:

- Host-retired actual candidate dispatch: only routed down submitted the
  candidate; native gate/up, route ordering/weighting and exact local output
  preserved. Original M2 split into M1 expert buckets stayed native. Payload
  weak references were dead, source file removed, and no artifact reupload.
- Host-retired typed Unknown after an actual successful candidate submission:
  no native retry, stop after first down, quarantine, operand/image retention,
  and no release on retry/quiesce/drop. This is injected Unknown custody testing,
  not a claim of a spontaneous hardware Unknown or a new poison test.

## Reduction-only follow-up is justified, not validated

`reduction_diagnose.py` performs CPU exact-rational arithmetic on the unchanged
FP8 bytes and BF16 scales/inputs at the two failed coordinates. It reproduces
the packed SIMT F32 FMA/warp-reduction schedule and both observed W8 words.
Both SIMT results lie exactly on their BF16 midpoint, so RNE selects the even
word, opposite the native/HF result.

- Column243: SIMT F32 is -0.00240325927734375. Exact dot is
  `-82575359/34359738368`, 2.9103830456733704e-11 above the midpoint. Correctly
  rounding this exact dot to F32 and then BF16 also gives `0xbb1e`: merely using
  a more accurate F32 sum does not guarantee HF equivalence (double rounding).
- Column1810: SIMT F32 is -2.3702159523963928e-7. Exact dot is
  `-16317/68719476736`, 4.220055416226387e-10 below the midpoint. Cancellation
  and reduction ordering can move the result across this BF16 boundary.

Sequential F32 accumulation matches HF at these two coordinates; an adjacent
pairwise F32 tree still fails column1810. These are diagnostics, not replacement
expected values or evidence that either schedule matches every output. A future
reduction-only candidate must retain identical decoded BF16 operands and rerun
all frozen words and existing tie/poison/Unknown gates before any larger run.
No backend kernel/reduction was edited in this task.

## Custody, pins and evidence

Evidence directory: `target/validation/mixed-routed-down-m1-small-20261010-02`.

| Artifact | SHA256 |
| --- | --- |
| `prepared.json` | `390bdfd71bdb2f4ff9d0ac19235da34ad34c4783ff32b93a88046ec10e105c2b` |
| frozen staged binary | `603b47a85122e40f9a98f147b967bb2247fb99d7795b6c0af7562081db740399` |
| model unit-test staged binary | `1415eaae3f438a472c22ed44fc247506c93fa7fdaf56deb657ca5f482520e8e2` |
| `result.json` | `735baa6aec360f32ad093a1176b1dae2cbba0128287bc78a3a397b0e78f79046` |
| `release.json` | `9e14b726df58d0541edac4a1fa8e274524db4439520f23935c95d5ed4f26d0fc` |
| `SHA256SUMS` | `cbd4d6ed217baeb88857f226c1412b57b16116b787b288d51575ccd660ed3413` |

The source/tool/shared-library pins were checked before and after execution;
read-only staged binaries were opened and hash-checked by descriptor. The
maintained authenticated supervisor completed in 15.04 seconds with
`proven_drained`, ECHILD and no cleanup errors. All owned birth identities were
subsequently absent. All eight devices were sampled idle before, after, and at
final release (zero memory/utilization, no compute apps). All eight shared local
locks were reacquired to verify availability, then released. No foreign process
was signalled. Resource samples are sparse small-test observations, not a peak
memory proof or performance acceptance.

`exact.json`, raw `rank*.{native,w8}.bf16`, logs, resource observations and
`reduction-diagnosis.json` are retained. `small_gate.py` owns only bounded small
tests and defaults to no GPU execution without an explicit grant argument.

The initial hidden build hit its 240-second deadline and was proven drained;
a same-limit retry completed using its cache. Post-run CPU tests passed (20),
CPU and CUDA12.4 targeted clippy passed with `-D warnings`, and the exact pinned
CUDA build environment passed `--no-run` again. One hand-assembled extra check
omitted the required native link paths and failed to find CUDA libraries; the
pinned-environment recheck corrected the command, not the code. Its receipt is
`target/validation/mixed-routed-down-m1-small-20261010-02-postvalidation.json`.
