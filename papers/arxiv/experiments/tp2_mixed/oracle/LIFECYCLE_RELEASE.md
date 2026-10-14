# Lifecycle CUDA reference release

Release SHA256: `834c8a1ff0aa12d0e58e500e870e68950a2dbe4e3cb6902d47fe02fdcd459639`.

All paths below are relative to this oracle directory. Machine-readable release:
`lifecycle-release.json`; frozen recipes: `lifecycle-contract.json`;
candidate schema and receipt rules: `BRIDGE.md`. No old reference, protocol,
run, numerical threshold or git index was changed by this release.

## Completed exports

| Directory | Calls | Manifest SHA256 |
| --- | ---: | --- |
| `runs/lifecycle-greedy4-01` | 10 | `4fafda33e51d7cec3d82beb59aac73eb12dc6b9904dc0a74535ea35e9d68df81` |
| `runs/lifecycle-fork2-01` | 6 | `098820b7839ed74b643066dcebd6e7e7b86d0716a8cbcecec708068d37b2a6dd` |
| `runs/lifecycle-canary2-01` | 5 | `4069b0e3f4dbead859654855a576c402867fa6ea2908d3dc67cf78c3ea2f672f` |
| `runs/lifecycle-recreate1-01` | 4 | `5747d1ee3793d629aff5d87488d2dbd970fb02be11742b88ac16e3dca05e5a59` |
| `runs/lifecycle-cold-owner1-01` | 4 | `fe7fb83bf8cae99164209aad077f5c1ff92adf902aaca43e8dbb99f8af8166ae` |
| `runs/lifecycle-cold-owner1-02` | 4 | `a67492ddb2a5a3d4a4f40abc1ca92c672f74605238d2f06ea62827f17cda3d54` |

33 calls / 38,577 reference tensors verified. Each call has 1,169 reference
and 1,089 required candidate tensors; the 80 omitted fields are the oracle-only
pre-tie native expert IDs. Every shape/dtype/finite value, per-tensor digest,
statistic, full-vocabulary rank gather and cache digest was checked using
existing CPU validators. Each export passed its authenticated 89-check CUDA
small gate before reading full checkpoint weights. All six children drained.

Thirteen reference/reference call-pair checks were byte-exact: cold-owner (4),
recreate (2), fork branch/unchanged-parent (2), canary/isolated greedy (5).
Greedy Hello/Capital prefill + first decode were also byte-exact against the
four corresponding immutable full40-03 stages. These are reference consistency
checks, not fabricated runtime candidate comparisons.

## Runtime replay

Use `runs/.../manifest.json -> suite.plan.operations` in order, including all
create/fork/drop/recreate operations. For each call replay the exact
`suite.events[].call.input_token_ids`, position/frontier and logical session.
`lifecycle-release.json -> runs[].calls` is a compact copy of these call records.
Each `call-NN.json` contains the exact one-call artifact inventory. Never feed a
divergent candidate argmax as the next input; compare its prediction separately.

- Greedy Hello inputs: `[9419]`, `[11]`, `[353]`, `[1044]`, `[264]`;
  next IDs: `11,353,1044,264,220`.
- Greedy Capital inputs: `[760,6511,314,9338,369]`, `[11751]`, `[13]`, `[198]`, `[760]`;
  next IDs: `11751,13,198,760,6511`.
- Fork: parent `[9419]`; clone left/right at frontier 1; left `[11]`, `[353]`,
  drop left; right `[11]`, forced `[760]`, drop right; unchanged parent `[11]`,
  then drop parent. The forced right token predicts `2614`; it is not greedy.
- Canary: main `[9419]`; canary Capital prompt then `[11751]`; drop canary;
  main `[11]`, `[353]`; drop main.
- Recreate: main `[9419]`, `[11]`; drop; recreate model; repeat in epoch 1.
- Cold-owner: Hello `[9419]`, `[11]`, drop; Capital prompt, `[11751]`, drop;
  identical recipe in two distinct authenticated processes.

**Runtime admission blocker:** its current contract admits only decode_steps=4.
Greedy4 can consume this release directly; fork2/canary2/recreate1/cold-owner1
need runtime-owner changes to exact plan fixtures and bounded budgets. The paper
contract will not silently expand to four steps, truncate, or change thresholds.
Use the runtime-owned lifecycle raw adapter, not the legacy two-stage bridge.
That adapter must preserve actual before/after state, receipt transaction/session
joins and shared-KV read-authority restrictions. No snapshot may be synthesized.

For the first compatible runtime launch, provide:
```text
FERRULE_MIXED_LIFECYCLE_ORACLE_DIR=papers/arxiv/experiments/tp2_mixed/oracle/runs/lifecycle-greedy4-01
FERRULE_MIXED_LIFECYCLE_ORACLE_SHA256=4fafda33e51d7cec3d82beb59aac73eb12dc6b9904dc0a74535ea35e9d68df81
```
The runtime owner must supply its own authenticated fresh launch/evidence and
GPU authorization; this release does not execute or authorize a runtime run.
After its CPU adapter emits a complete canonical candidate, use the unchanged
GPU-hidden `lifecycle_compare.py` CLI in `BRIDGE.md`, with both manifest pins.
Acceptance requires every required tensor and unchanged tolerance
`0.00025 + 0.012 * abs(expected)`; selected IDs/inputs/greedy/frontiers are exact.

## Numerical identity and scope

Pinned Torch `2.13.0+cu129`, Transformers `5.5.3`, vendor HF module source hashes
and live interception audit are in each manifest. TF32 and BF16 reduced-precision
reduction are off. CUDA reference uses logical TP2 on one RTX 3090, native HF
forwards plus partitioned operands, original FP8 payload/scales decoded to BF16,
explicit F32 SSM and ascending-expert HF eager accumulation. No production
model kernels or runtime observations supplied expected values.

Checkpoint: 64,196 tensor headers; 37,463,724,680 whole-shard bytes; shard-set SHA256
`b51014a9d40c300ea77f12bf8cada444f79c8300277611f6e5f27a62fcb95090`.
Original names, sizes, dtypes, offsets and file hashes are pinned per run.
Protocol SHA256 `c6e7836f3797c16502dc21b8c756c87495525c2cf2faac26ba11077f62cfbb57`;
full40-03 manifest SHA256 `fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d`;
lifecycle contract SHA256 `684dd2f85798745178a3a6719daaf23cc44471072fb3c35e3155065fb8ba170d`.
These original pins still match; old sealed source pins are not rewritten.

Unresolved production equivalence sites remain explicitly listed in the release:
CUDA GEMM/norm/softmax/activation/reduction implementation differences; GDN chunk
prefill versus recurrence, exp/softplus/L2 and exact F32 state updates; HF eager
per-expert accumulation versus vLLM fused MoE. The old two-stage full40 PASS does
not establish new multistep runtime acceptance. No CPU/CUDA exp equality, vLLM
bit-equivalence, physical TP2, performance or production COW/cancellation claim.

## Validation and release

GPU-hidden CPU tests passed with the required separate interpreters:
```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda310/bin/python -B -m unittest test_tp2_oracle test_compare test_multistep test_bridge test_lifecycle_compare
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest test_launch
```
Run from this oracle directory: 46 numerical/bridge/lifecycle + 5 launcher tests.
Initial unified discovery used the wrong interpreter; split commands above passed
without weakening the base-Python authorization check. Final artifact validation
was CPU-only (`torch.cuda.is_initialized() == False`), not reference generation.

All eight GPUs were 0 MiB / 0% with no compute processes at release. Every owned
process birth identity was absent; the existing local owner lock was verified
available and immediately unlocked. GPU ownership is released. No foreign PID
was signalled. Subsequent GPU work needs coordination with the runtime owner.
