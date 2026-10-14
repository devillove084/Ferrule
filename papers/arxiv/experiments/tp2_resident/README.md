# TP2 resident vLLM experiment

This is a new, separate paper experiment. It does not edit or reuse the old
M7/G10 manifests as a TP2 claim. `experiment.py` is the first observer revision;
`experiment_v2.py` adds bounded `ESRCH` resampling with stable live-root identity.
Runs `tp2-resident-20261008-001` and `tp2-resident-20261008-002` are immutable evidence.

## Profile

- Qwen3.5-35B-A3B-FP8 at `/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8` (text-only)
- vLLM `0.28.0+cu129`, Torch `2.13.0+cu129`, CUDA 12.9 JIT allowlist
- TP2 / PP1 / DP1, **no expert parallel**; TP2/EP2 is a future separate profile
- no CPU offload: explicit `--cpu-offload-gb 0`; installed-source defaults verify
  `offload_group_size=0` and `NoopOffloader` selection
- BF16 model/activation/KV, `auto` KV resolution, BF16 Mamba cache, F32 Mamba SSM
- Marlin linear and MoE backends, language-model-only, prefix cache off, eager,
  seed 1, max model length 128, max batch tokens 8, max sequences 2, chunked prefill
- vLLM planning: `0.8 * 24 GiB = 19.2 GiB/card`; observed sampled ceiling 22 GiB/card
- host process-tree PSS budget: 128 GiB, a declared common planning budget rather
  than a physical hard cap
- explicit `NCCL_P2P_DISABLE=1`; historical source evidence and live logs show
  P2P `CNS` / SHM transport. Transport is recorded, never silently substituted.

Preparation seals source, staged entrypoint, installed runtime/shared-library/JIT,
model metadata/shard-stat identity, reference manifest, and helper pins. Payload
hashing of the 33.26 GiB checkpoint is intentionally not performed; this remains
an explicit partial identity. Dynamic driver/JIT closure remains unknown.

## CPU-only checks

```sh
env CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=none \
  /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  -m unittest discover -s papers/arxiv/experiments/tp2_resident -p 'test_*.py'
```

## Bounded GPU lifecycle

```sh
env HOME=/home/admin CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=none \
  /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/tp2_resident/experiment_v2.py prepare --id <new-id>

env HOME=/home/admin CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=none \
  /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/tp2_resident/experiment_v2.py run --id <prepared-id>
```

The runner acquires sorted physical UUID locks for GPU0/GPU1, requires complete
all-eight idle samples before and after each finite case, authenticates the local
subreaper descendant drain, and never signals a foreign PID. It runs Hello2,
then independent Japan/arithmetic/Chinese eight-token visible-prefix canaries,
then a separate descriptive QA block only after canary success. It never claims
exact arithmetic or speedup.
