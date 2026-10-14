# Routebatch ordinary-info actual result

Binary `e674eda31c0b1379e19afbd1446662278d98047c27b77c0f1233670e601c87f7`; same-binary S1/S2 cancellation/reuse/normal-shutdown gate PASS, excluded from performance.

5 measured requests, each 5-in/8-out: 40 output tokens / 2.811538 s = **14.227089 token/s**.
TTFT mean 0.178214 s; E2E mean 0.532167 s.
Startup 351.114481 s; lifecycle / supervisor / outer stage wall 365.922026 / 400.686129 / 454.722097 s.

Existing fused_after/F1, case optimization=routebatch; fresh Hello/canary then excluded-first + warm2 + measure5. Ordinary info, rankperf off, no V/Nsight/readback/matrix.
TP2 thread, ctx128/batch8/seq2; arena3 GiB/workspace4 GiB/device22528 MiB. Physical/fallback cache-ledger fields are retained in JSON; activity and auxiliary kernel count are null without official profiling.
All eight GPUs 0 MiB and empty compute roster; UUID locks reacquired/validated/released; authenticated drain and unchanged source pins verified.
Host default/NCCL disabled, private host/device bounds Unknown. No internal-2178 or independent real-model R64 acceptance claim.
Historical f6a 6.5148, old7e1 5.7, V10.5 token/s are descriptive only: no causal/statistical/exact-speedup claim.
