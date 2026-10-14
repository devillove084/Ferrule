# Routebatch versus frozen vLLM: two descriptive crossblocks

Order F1→V1 / V2→F2; same selected two GPUs; four fresh owners; prior actual same-binary smoke reused. No Nsight, new matrix, package upgrade, Marlin downgrade or tensor readback.

| Slot | Window s / 40 outputs | token/s | TTFT mean ms | E2E mean ms | Startup s | GPU peak GiB/card | Host PSS peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1 | 2.793861 | 14.317105 | 176.248 | 523.399 | 349.585 | 21.041992 | 61.046329 |
| V1 | 3.854979 | 10.376192 | 132.676 | 694.228 | 66.239 | 19.284180 | 7.711452 |
| V2 | 3.872315 | 10.329739 | 134.651 | 699.213 | 62.814 | 19.284180 | 7.323862 |
| F2 | 2.701918 | 14.804301 | 171.333 | 515.106 | 366.133 | 21.041992 | 60.137685 |

Block A F/V output-rate ratio: 1.379803; descriptive only.
Block B F/V output-rate ratio: 1.433173; descriptive only.

Each slot: Hello/canary, excluded first QA8 +warm2, then measure5; actual each 5-in/8-out.
F completed this short fixed-work load faster in both repetitions. V had lower TTFT, much shorter startup and lower host PSS; not a universal performance claim.
Ferrule FP8 checkpoint/BF16 activation-KV-conv/F32 SSM, HF-eager GemmEx/routing source qualification; vLLM 0.28.0+cu129/Torch2.13.0+cu129, BF16/FP8 Marlin. No arithmetic or internal-state parity claim.
Ferrule workspace4 GiB incl arena3 GiB/device22 GiB; frozen V utilization0.8 (19.2 GiB planning on24 GiB). Common sampled22 GiB ceiling is not identical total-memory flags/hard partition.
Same checkpoint/tokenizer/URL; frozen engine-specific served-model aliases differ and are retained, explicitly recorded.
Fresh V logs show CUDA graph mode NONE and prefix caching false; Marlin runtime gates passed. Existing sampled F model-process maps contained no libnccl; V maps contained pinned libnccl.so.2. Ferrule private bounds remain Unknown; no host-pipe budget proof for NCCL.
Observed memory peaks are sampled whole-device/server-tree values, not exact peaks or per-process GPU attribution. Activity/auxiliary kernel count null without official profile.
All eight GPUs0 MiB/empty compute roster; UUID locks reacquired/validated/released; all four authenticated descendant drains and source/runtime/API pin checks passed.
n=2: descriptive ratios only; no statistical significance, causal speedup, fairmatch/arithmetic parity or PAPER E2E checkbox PASS.
