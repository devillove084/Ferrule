# Strict TP2 startup_metadata completed; GPU ownership released

Exact sealed binary `36e6aaa54d1aff7c8cc35d63dcee0df1cad91fcc8f6d3e746369504f2c70a58c`; no rebuild.

| Lifecycle | Startup s | Sampled peak tree PSS GiB |
| --- | ---: | ---: |
| S1 | 222.079 | 42.716 |
| S2 | 139.855 | 44.416 |
| F1 | 140.052 | 43.573 |

S1/S2 fresh owners passed visible greedy canary/reuse/cancel/normal shutdown. Only F1 ordinary info: 5-in/8-out, 3 excluded QA requests (first + 2 warm), 5 measured, 25 input/40 output; 2.877869 s measured, 13.899175 length-matched usage output tokens/s. TTFT mean/p50/p95 192.944/187.513/209.700 ms.

All three actual /proc/exe hashes and all 12 runtime library maps passed; NCCL absent. Five F1 proc reads are verified outside measurement. Costs/Anonymous/RssAnon/Threads/CPU masks retained; shutdown-after-exit unavailable/null; Pss_Anon/Pss_File unavailable/null. No heap-live inference. Both GPUs sampled peak 21547 MiB against unchanged 22528 MiB paper target, not hardware capacity or continuous hard-cap proof.

Preserved e674 context remains historical and non-controlled; no fresh BEFORE/de288 measurement. Current optimizations affect CPU metadata, not engine mathematics. Inherited 14M logits are not new numerical validation. No V/Nsight/storage TP1/additional pairs ran. Tests passed 23 paired +22 observer +15 smoke policy +8 release policy +7 fixed-case checks; builder receipts retain 111 CPU passed/4 ignored. Initial schema/test failures remain recorded. Exact executed smoke/ordinary Python+config revisions were recovered and hash-verified, not relabelled. All supervisors proved authenticated ECHILD; UUID locks released; final eight GPUs idle. Ownership released; next single-card storage owner awaits main authorization.
