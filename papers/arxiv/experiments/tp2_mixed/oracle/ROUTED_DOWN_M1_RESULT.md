# Routed-down M1 independent CUDA fixtures — completed

## Release identity

Root: `runs/routed-down-m1-cuda-01`.
Manifest SHA256: `4f3081e16cc496121817bd49e4732cb5b49a030ca9853f616751c17d23ba1704`.
CPU validation and final GPU release: `runs/routed-down-m1-cuda-01.validation.json`.
Validation SHA256: `f8ebffdea73c4f0ed4825d4e93e8327884bb6630d01da8c7cf6a9b6fbbf9afb0`.

Every child below contains `manifest.json`, the eight required raw files and
`SHA256SUMS`. Use the **child directory** for `FERRULE_MIXED_DOWN_FIXTURE` and
its **SHA256SUMS hash** for `FERRULE_MIXED_DOWN_SHA256`; the release manifest hash
is not the Rust fixture pin. Paths in this table are relative to the root above.

| Fixture | SHA256 of SHA256SUMS |
| --- | --- |
| `hello-decode0-layer00-expert021` | `e2bbcee5f8cd281b21bc87eea81663e739e8ab095a744b59ba2ddaf19365291f` |
| `hello-decode0-layer03-expert037` | `985454817c99304845b479b3697f45581d38b4b7910de277836cd98102128b21` |
| `capital-decode0-layer00-expert072` | `23f2715bea81a25f253d386e02168ab130dc7d41131dcb3662c7fb86078ef575` |
| `capital-decode0-layer03-expert010` | `f8446da2922dd38ce67e4941f3ac9a6dcb9d86cb39ec2699492e4c58c9c8c3f4` |

## What actually ran

- Fresh authenticated small-task adapter `routed_down_launch.py`; existing
  paper seal/authentication/lock/subreaper and `tp2_oracle.authenticate_export`.
- One CUDA GPU, two logical rank branches; no full40 forward or new state update.
- Four archived decode case/layer replays, eight HF `Experts.forward` calls,
  64 actual expert-down projections. Four selected experts, both ranks: eight
  captured SwiGLU input rows and eight pre-route-weighting BF16 output rows.
- All 40 controls byte-exact against immutable CUDA observations: raw router
  logits, F32 probabilities, selected IDs, route-weight words, and weighted
  eight-expert aggregate, each for all eight case/layer/rank combinations.
- N=2048, global K=512, local K=256; BF16 scales 16x2 per rank. The raw FP8
  and scale bytes are identical to the authenticated CPU extraction slices.
- Torch 2.13.0+cu129 / Transformers 5.5.3; pinned native HF source. Native
  BF16 residual/norm/router and unchanged HF expert forward, existing original
  payload decode/partition helpers. F.linear observation only captures actual
  arguments/results before weighting and is restored/audited afterward.
- TF32 and BF16 reduced-precision reductions off; oracle CUBLAS setting
  `:4096:8`. No candidate kernels, observations or tolerances supplied expected.
- Child timeout 120 seconds; outer invocation 180 seconds; 16 MiB capture cap.
- Actual numerical time 4.181s; supervisor 6.407s;
  captured 36,864 bytes; peak Torch reserved 120 MiB.

## CPU checks and handoff

Four new launcher tests passed with the base Python -S: no-I/O preflight,
missing grant rejection, strict bounds/pins, free-worker rejection. The six
CPU extraction tests passed in the prior preparation. Dependency/source/bundle
pins and HF ExpertBank binding were verified GPU-hidden before the CUDA launch.
After generation every consumed file hash/size, BF16 finite word, FP8 finite
byte, scale positivity and original payload identity was verified on CPU.

Rust CPU preflight was not executed by the oracle owner: no prebuilt
`mixed_routed_down_frozen` binary was found under the local target tree. No
new build tree or model/runtime source edit was introduced. The model owner can
run its existing CPU-only test against each child. First fixture example:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
 FERRULE_MIXED_DOWN_FIXTURE=papers/arxiv/experiments/tp2_mixed/oracle/runs/routed-down-m1-cuda-01/hello-decode0-layer00-expert021 \
 FERRULE_MIXED_DOWN_SHA256=e2bbcee5f8cd281b21bc87eea81663e739e8ab095a744b59ba2ddaf19365291f \
 cargo test -p ferrule-model --test mixed_routed_down_frozen frozen_tp2_down_fixture_preflight -- --ignored --exact --nocapture
```

**Native and W8 GPU operator gates have not been run.** They require a separate
owner handoff and use the model test contract, including its own `:0:0` workspace
setting. If either differs from HF words, report the actual mismatch; do not
adjust the fixture, relax exactness, or replace expected with candidate output.

## GPU released and immutable inputs

Custody proven drained; eight GPUs sampled idle before/after and at final release,
0 memory/0 utilization and no compute apps. All owned process birth identities
were absent; the shared local owner lock was verified available then unlocked.
No foreign process was signalled. GPU ownership is released.

Old full40-03 reference, numerical protocol, CPU extraction bundle and existing
launcher/oracle sources were not edited. Fresh source pins, original payload
identities, archived call/route provenance, scale-block coordinates and actual
capture authentication are embedded in each fixture manifest and the release.
No physical TP2 transport, production quality or performance acceptance claim.
