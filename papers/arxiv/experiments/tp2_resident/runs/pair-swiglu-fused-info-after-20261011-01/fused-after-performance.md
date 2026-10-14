# Fused-after ordinary-info result

Status: complete_verified; fixed binary `f6a65e1b6f09c0a8385fcfb0ee0b1cab5f02d2b09a341a6cd4d9e7aa81ed06f4`.

| Metric | Actual |
|---|---|
| Measured requests | 5; each 5 input / 8 output |
| Output tokens / measured window | 40 / 6.139863 s |
| Output token rate | 6.514803 token/s |
| TTFT mean (range) | 0.441015 s (0.435735–0.448332) |
| E2E mean (range) | 1.201992 s (1.194183–1.214975) |
| Startup | 348.655296 s |
| Lifecycle / supervisor / outer stage wall | 371.324439 / 401.650076 / 449.249669 s |
| Auxiliary kernel count | null: no official profile |

One fresh F1 lifecycle: Hello/canaries, excluded first QA8 + warm2, then measure5. `FERRULE_LOG=info`; no rankperf, V, cross-AB, Nsight or tensor readback. Actual S1/S2 request-debug smoke PASS is separate and excluded.
TP2 thread; 3 GiB arena, 4 GiB workspace, 22528 MiB device cap; context128/batch8/sequences2; BF16 activations/F32 SSM.
All eight GPUs have 0 MiB used and empty compute roster. Existing UUID locks were reacquired/validated/released; authenticated descendant drain, normal shutdown and unchanged source pins verified.
Historical BEFORE 7e1 5.66–5.82 token/s and V 10.5 token/s are descriptive references only. No pure causal effect, statistical significance, exact speedup, internal-state equality or kernel-count reduction claim.
