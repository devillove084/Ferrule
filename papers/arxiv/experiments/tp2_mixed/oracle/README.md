# Independent TP2 mixed oracle

Paper-local reference protocol for `SystemMixedTP2EP1`, separate from historical
unsharded HF captures. **A completed CUDA reference is available:**
[`runs/full40-20261009-03/HANDOFF.md`](runs/full40-20261009-03/HANDOFF.md).
It passed 89 small CUDA checks and full artifact integrity verification. This is
not a Ferrule full40 comparison, factory admission, physical TP2 result, or
performance acceptance. Failed attempts 01 and 02 remain preserved.
All commands below run from the repository root. New assets stay in this folder;
the original reference scripts are unchanged.

## Numerical contract

- Torch `2.13.0+cu129`, Transformers `5.5.3`, and the three original HF source
  pins are enforced. Live forward/helper source and canonical bytecode/constants hashes are recorded.
  Raw marshal hashes are diagnostic only: Python object-sharing changes their bytes.
- Original E4M3FN payloads and numeric BF16 128×128 scale grids are decoded in
  F32 then rounded to BF16. Shard coordinates come from independent config
  formulas, with global block coordinates and local decoded operand checksums.
- Published HF forwards own norm, RoPE, attention, GDN, experts, decoder order,
  and cache updates. Only operand-backed projections, rank wrappers and the
  raw-tie router hook are replaced. No Ferrule/vLLM kernels are imported.
- Attention shards whole query+gate and KV heads; GDN shards Q, K and V
  segments separately, with matching conv channels, Z/B/A and state heads.
  Local output-projection partials are BF16; rank 0 + rank 1 is computed in F32
  and rounded once to BF16. TF32 and BF16 reduced-precision reduction are off.
- Each rank has access to a slice of **every** expert. Lazy checkpoint loading
  and a bounded decoded-operand cache are allowed, not measured residency or
  offload performance. Gate/up halves are sharded independently before concat.
- Router weights are replicated BF16; native probabilities and routing weights
  are F32. Lower-ID substitutions occur **only at equal raw logits**, not at
  unequal logits whose softmax probabilities happen to collide.
- Routed accumulation is **HF eager ascending expert ID**: each F32 weighted
  expert output is rounded to BF16, then accumulated in BF16. Native shared
  sigmoid gating and routed+shared BF16 addition happen **locally before one
  TP sum**. This is not slot-major or vLLM fused-MoE accumulation.
- Embedding and final norm have replicated semantics. BF16 vocabulary row
  shards are concatenated in rank order, without reduction. GDN state is F32;
  conv and KV caches are BF16. Both local and reconstructed caches are captured.

Two logical TP ranks execute on **one granted CUDA device**. This tests the
numerical partition/reduction protocol, not physical TP2/NCCL transport,
performance, or residency. There is no vLLM bit-exact claim and no CPU/CUDA
exponential or state-equivalence claim.

`protocol.json` names remaining math-site risks: CUDA GEMM/softmax/norm/activation
implementation drift, HF GDN chunk prefill versus production state updates,
exp/softplus/L2 normalization, and vLLM fused MoE's different order. Uncovered
sites or failures block runtime admission; they never justify silent tolerance
changes. State comparisons report exactness **and** the unchanged tolerance.

## CPU validation and preflight

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
  /opt/conda310/bin/python -B papers/arxiv/experiments/tp2_mixed/oracle/test_tp2_oracle.py
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
  /opt/conda310/bin/python -B papers/arxiv/experiments/tp2_mixed/oracle/test_compare.py
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
  /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/tp2_mixed/oracle/test_launch.py
```

Tests cover TP1/native prefill+decode equality; TP2 partition/rounding witnesses;
GDN groups/cache assembly; gate/up concat; expert accumulation and raw ties;
FP8 global scale coordinates; complete tiny-model serialization; drift mapping;
unchanged tolerances; missing/stale/nonfinite/misordered artifacts; and
fail-closed launch/child authorization. Fixtures use tiny random CPU tensors,
not original model weights. The custody probe uses the existing bounded helper.

A CPU preflight hashes scripts/protocol but does not open the model or query GPUs:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
  /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/tp2_mixed/oracle/launch.py \
  --output papers/arxiv/experiments/tp2_mixed/oracle/preflight-YYYYMMDD
```

Without `--output`, it prints the same record. Output directories must be fresh.
A preflight is **not** permission to export, nor proof the real model fits.

## Authorized full40 export

Only after a fresh explicit sole-owner grant, the GPU owner supplies the actual
UUID and note to `launch.py`. It reuses paper `contract.py`, `launch_guard.py`,
telemetry, and `test_vllm_qwen35.run_process` / `_owned_subreaper.py` (custody
utilities, not vLLM model code). The shared paper lock is the one intentional
existing lock namespace outside this folder; it must not be replaced with a
private lock that could bypass another owner. No foreign PIDs are signaled.

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/tp2_mixed/oracle/launch.py \
  --output papers/arxiv/experiments/tp2_mixed/oracle/run-YYYYMMDD \
  --execute-cuda --sole-gpu-owner-granted \
  --gpu-uuid ACTUAL-GRANTED-GPU-UUID --owner-note 'Actual fresh owner grant'
```

This is a template, not an authorized command. The default model comes from the
existing pinned exclusive paper configuration; `--model` may override it with
the original full40 FP8 checkpoint. The numerical Python defaults to
`/opt/conda310/bin/python`; its executable hash is sealed too. The child verifies
its live authenticated worker/helper/coordinator ancestry, launch receipt,
token, PID namespace and owner lock **before checkpoint reads or CUDA queries**.
Direct `tp2_oracle.py` execution fails closed.

Bounds: `--timeout` at most 1700 seconds, `--cache-mib` at most 1024 (default 256),
`--capture-mib` at most 2048 per case (default 512). Source hashing and export are
inside the supervised deadline. Resource/timeout failures remain incomplete.
All original indexed shard files are hashed during authorized export, while
`source-payloads.json` separately records actually used raw operands/scales.

Cases are the original token sequences: `hello=[9419]` and
`capital=[760,6511,314,9338,369]`. Each has prefill plus `decode.0`, fed from the
oracle's own full-vocab greedy prefill ID. The consumer must replay that recorded
ID, not substitute its own continuation.

Optional `--unsharded-reference DIR --unsharded-sha256 SHA256` records original-HF
max/mean absolute error, RMSE, exactness, unchanged-tolerance failures, route
changes and full-vocab argmax agreement. If cached inputs differ, decode drift
is explicitly not compared as same-input. This is descriptive quality drift,
not acceptance against the old unsharded reference. Without a supplied baseline,
the report says `not_supplied`; old reference directories are never modified.

## Artifact handoff and comparison

A successful export contains:

- `hello.safetensors`, `capital.safetensors` and corresponding JSON reports:
  every layer/stage, per-rank attention/MoE/shared/routed partials, router
  logits/probabilities/weights/IDs, local and reconstructed state/KV, embeddings,
  final norm, local and full-vocabulary logits, inputs/positions/masks.
- `operand-layout.json`, `source-payloads.json`, pinned source/interception and
  whole-checkpoint identities in the manifest.
- Sealed launch, worker and numerical-child authentication, before/after idle
  telemetry, and bounded proven-drained supervisor receipts.
- `export.json` (numerical completion pending custody), then `manifest.json`
  and `manifest.sha256` (final coordinator decision).

`status=complete` and `full_reference_complete=true` mean **reference export and
custody completed**, not Ferrule validation. `strict_ferrule_full40` remains
false and `runtime_gate.status` remains `not_run`. An absent/incomplete manifest
is never an admissible reference. The completed reference and preserved failures are linked in the handoff above.

`compare.py` defines an explicit **candidate interchange contract**; it is not
an already-wired Ferrule runtime exporter. The main runtime gate must supply
both directories and independently pinned manifest SHA256s:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /opt/conda310/bin/python -B \
  papers/arxiv/experiments/tp2_mixed/oracle/compare.py \
  --reference REFERENCE_DIR --reference-sha256 REFERENCE_SHA256 \
  --actual CANDIDATE_DIR --actual-sha256 CANDIDATE_SHA256 \
  --output papers/arxiv/experiments/tp2_mixed/oracle/comparison-YYYYMMDD.json
```

Candidate manifest schema is `ferrule.paper.tp2-mixed-candidate.v1`, with
`profile`, `status=complete`, the exact protocol, matching checkpoint
`config_sha256`, `index_sha256`, `full_shard_set_sha256`, and
`effective_text_config`, plus `cases` in `hello,capital` order. Each case has
`id,file,sha256,report`; `report` is a `{file,sha256}` record. Reports contain
`id`, `calls` and `tensors` (shape/dtype inventory). Calls contain `stage`,
`input_token_ids`, `position_start`, `sequence_length_after`, `next_token_id`.
Use `candidate_inventory(config, ids)` for exact capture names/shapes/dtypes.
Only HF pre-canonicalization `native_expert_ids` are oracle-only diagnostics;
canonical `selected_expert_ids` and **all** other boundaries are mandatory.

Comparison rejects stale hashes, missing inventory, wrong geometry/dtypes,
nonfinite values, wrong prompts/frontiers/gathers and fabricated reported
argmax. Numeric tolerances remain `abs(actual-expected) <= 0.00025 +
0.012*abs(expected)`; IDs/inputs are exact, and greedy IDs must match. A report
is written on numeric pass/fail; malformed artifacts raise a structural error
and produce no passing report. Candidate production execution/custody and
factory admission are separate obligations: neither copied oracle tensors nor
a synthetic comparator PASS proves them. Adapter availability and uncovered
math-site differences remain blockers for the full runtime gate.
