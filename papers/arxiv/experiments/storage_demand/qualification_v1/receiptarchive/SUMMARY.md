# Current expert-read real-GPU qualification

**Result: 9/9 exact-once GPU gates passed on committed `033519c`.**

No probes, warmups, retries, profiling captures, production/paper changes, commits, or pushes were made by this qualification. Prior `-01` artifacts remain untouched.

## Artifact identity

| Artifact | SHA256 |
|---|---|
| cli | `de5159951fe724e97e35c0fc86367642d33735994328eee1a5e604cf9d2b037b` |
| constructor | `cb9b8c998449b8f73a6a6919be19bcd8e1d743711a1c06d36e5332af57db4a23` |
| composition | `8810c3418f4b39b4c9203003b91e53a6106a833624db4d13a516fdfd63453d6a` |
| Source archive | `ac072e0d1bac7efcc85b7f26ce68723f412b0e22b5ee868104b1f6073dd2a82f` |
| Absolute-path source pins | `a4dceb084e21b80f6b45ff7ac980c6fa80ac8f17e902fd28c65f3d5bc7bb0efa` |
| Relative source-member pins | `bde8b14396855df4cf832cb89d01250f2b4e162ef7ed9b3abc93d2c4df975fa7` |

CUDA 12.4.99, SM86, CUTLASS `e05f953a5b3d38adc240df2ff928e0421c2abba3`, official installed NVTX3 headers; common NVTX feature compiled and `FERRULE_PROFILE_NVTX=0` at runtime. CLI/constructor/composition are distinct executables; common source/native/runtime provenance is not a mathematical binary-equivalence proof.

## Actual gates

| Gate | Result | Main report metrics | Main logit values | Actual fingerprints | Report SHA256 |
|---|---|---:|---:|---:|---|
| 01-constructor | PASS | 0 | 128 | 0 | `bea993f0f42c09e61bf10ebd8944aedc8ce6c8c29af99f999ddb3aca35488789` |
| 02-blocking-full40 | PASS | 804 | 1986560 | 0 | `bdad062495ab3ab12792c89fa45f82b7405a9d9bec46fd40302656adbedc85dd` |
| 03-lookahead-full40 | PASS | 804 | 1986560 | 0 | `96de433f84f08c24affabec49d0261b208231233060cdaff2d3691e53fd93283` |
| 04-blocking-fresh | PASS | 5427 | 9187840 | 64 | `0b3ad18b6ade1a63b35db0a86bb5473350617e63533ff23fba8e1da9a16a6160` |
| 05-lookahead-fresh | PASS | 5427 | 9187840 | 64 | `8063e1c747aff46da1493f4151f2053a3f860d230ffac153eff79af8c448228f` |
| 06-blocking-ordinary | PASS | 27 | 9187840 | 64 | `dd566d7c4c4cbad6d1bb838c6df3eec0492d3912f56b53418285c504f7be17a1` |
| 07-lookahead-ordinary | PASS | 27 | 9187840 | 64 | `8602f3c286b483dc7a891d79f37987aafd1626436b4e8cc832e8bfd0acd36f29` |
| 08-blocking-cold | PASS | 1809 | 3228160 | 28 | `bb4d65a2df3d09838216c47a6d1862bf7ea9f06095a5350ec4bdecf41f028089` |
| 09-lookahead-cold | PASS | 1809 | 3228160 | 28 | `4035d9f48c6387d82ca36a0a3937dd365b15cad42f56b34ed49811be8b925397` |

Constructor: four boundary/schedule cells, 128 nonempty finite logit values, six verified actual reads per cell. This is tiny constructor/provider evidence, not full-model oracle acceptance.

Composition capacities: host64MiB including metadata8MiB/stack2MiB; device64 entries/cache320MiB including scratch8MiB; device7GiB/workspace2GiB/KV16MiB; context128/batch8/prefill8/sequences2. Observed allocator peak4,106,223,616 bytes; logical host charge peak16,777,984 bytes. Neither is a certified physical RSS/release claim.

Both ordinary runs exactly match all 64 same-schedule diagnostic actual logit-word/shape/greedy fingerprints, with their current diagnostic primary SHA pinned. Reference checks retain `0.00025 + 0.012 * abs(reference)` and exact routes/greedy IDs. Cold owners bind those same current primary configs/oracles.

Per fresh/ordinary schedule: common-history27 calls/9,187,840 values; free-feedback27/9,187,840; reset9/3,228,160; fork1/248,320. Per cold schedule: common-history9/3,228,160; free-feedback9/3,228,160; reset9/3,228,160; fork1/248,320. Streams are not summed into a misleading single denominator.

Final cleanup: every process exited naturally; authenticated maintained-subreeper ECHILD, absent recorded PID/birth owners, all-eight idleV2 checks; final three zero-memory/zero-utilization/no-app samples. All eight local UUID locks released and successfully nonblocking reprobed. Physical CUDA release remains unobserved.

## Receipts

- `FINAL.json`: coordinator outcome and all gate results
- `receipt-audit.json`: independently cross-checked identities, hashes, denominators, primary bindings, mapped runtime pins, resource observations and limitations
- `qualification-plan-v2.json`: final prelaunch immutable plan; earlier prelaunch plan/adapter preserved
- `source-freeze.json`, `archive-exact-members.json`, `source.tar`: exact 1,484 committed source members
- `gates/<gate>/`: one-test list, sealed worker dispatch, authenticated launch, stdout/stderr, typed quality report, supervisor custody, three idle receipts, actual PID/birth runtime mappings
- `final-idle.json`, `final-lock-release.json`, `final-lock-probe.json`: final cleanup observations

No failure stop reason: all nine gates completed. Historical CUDA-driver flakiness remains unexplained. Concurrent paper edits are outside this qualification and were not modified.
