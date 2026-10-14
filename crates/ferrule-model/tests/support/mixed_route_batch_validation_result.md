# Incremental routed batch — GPU and E2E validation

2026-10-12. **Ready for release-builder handoff for the validated default host-control path.**
No production/runtime/oracle changes were made by the validator. No new release
binary, simple release E2E, or performance measurement was produced.

## Receipts and source boundary

- Backend and initial five model gates plus seven E2E gates:
  `papers/arxiv/experiments/tp2_resident/runs/route-batch-20261012-02`.
- Final model gates with explicit true H2D/launch counter assertions:
  `papers/arxiv/experiments/tp2_resident/runs/route-batch-20261012-04`.
- Final handoff: `target/validation/route-batch-incremental-20261012-01/ready-release-builder.json`.
- Handoff SHA256: `4d870c9aadd427fc5a52eda03de474ed0b5bc9c01e5f1a2f9741810115430b0b`.
- Same directory: `default-off-audit.json`, `gpu-release.json`, `final-idle.json`,
  compiler/linter/CPU receipts and original test/workflow archives.
- Additive receipt: `mixed_route_batch_validation.sha256` in this folder.

All GPU selectors resolve to exactly **one ignored test** and ran under the
existing authenticated supervisor/subreaper, fresh selected UUID locks and
matching visibility. Pinned tao CUDA 12.4.99/sm86, original tool inventory,
`CUBLAS_WORKSPACE_CONFIG=:0:0`; each compiler/link/lint bound was 600 seconds.
No extra GPU synchronization or replacement execution loop was added.

The original `mixed_route_batch.sha256` was preserved, not rewritten. Its
original GPU-test module is archived and verifies against its original pin.
Validator changes were confined to tests and the existing paper runner:

1. Add two actual model GPU gates: late consumer-proof Unknown and equal-shape
   foreign model/source rejection before payload allocations/H2D/kernels.
2. Strengthen the real partial gate with actual copy/launch counters.
3. Fix one test-only `RoutedOutputLayout` import revealed by CUDA test linking.
4. Add a pure CPU synthetic R64 gather/metadata budget-floor test.
5. Add a narrow route-batch selection to the existing paper launcher; reuse its
   locks, observer, supervisor and existing full-vocabulary comparison.

E2E/current source inventories differ only in the GPU-unit counter strengthening
and the isolated CPU R64 budget test. Both are explicitly recorded; the former
E2E unit source is archived. Production and composition executor/test sources
are identical between E2E and the final inventory. Do not describe this as one
unchanged whole-repository snapshot.

## Actual numerical and lifecycle coverage

| Gate | Executed result |
| --- | --- |
| Backend old per-expert versus batch gather/combine | Exact u16; M1/M5 overlap/disjoint, R40 and R64, offset canaries |
| Backend owners/aliases/preflight/post-H2D fault | Passed; same-ordinal different owners, no invalid preflight work, no ready publication after fault |
| Backend finite F32 bits | Passed; original weight bits retained, nonfinite metadata rejected |
| Model old independent routing versus batch local partial | Exact raw u16; both lanes, M1/M5, overlap/decode, native/cached cold/hot, retired sources and removed fixture file |
| Model actual gather Unknown | Metadata host/device allocation and gather root retained; cache on/off, no replay or release on owner drop |
| Model native diagnostic observer | Complete expert input views at actual offsets into one gather root; same batch gather, old diagnostic pair tail |
| Model late existing consumer-proof failure | Actual batch submissions followed by failed completion proof; metadata/gather/output/image backing retained, no replay |
| Model equal-shape foreign source | Public preparation rejected before payload allocation, H2D or kernel submission; no image/provider publication |

The foreign-source gate does not claim that creating an authorized CUDA owner or
its pre-existing cleanup fences performs zero driver API calls.

The final partial gate exported **32 actual counter rows**. For E selected expert
buckets, the unchanged legacy routing uploads E gather ID sets plus combine IDs
and weights; batch routing uploads one opaque allocation:

- H2D calls: **E+2 -> 1**. Observed E2 cases: 4 -> 1; E1: 3 -> 1.
- H2D bytes: **12R -> 8R**, asserted on actual operator counters.
- Compute kernels: legacy minus batch = **E-1**, with identical partial words.
- E2 cached hot: 9 -> 8 compute kernels; cached cold: 15 -> 14;
  encoded native source path: 19 -> 18. These are launch counts, **not latency**.

## Existing cache-on, no-observer E2E

No history/2178-boundary capture was installed. The unchanged composition
executor ran `Some(3 << 30)` decoded cache, `trace=None`, zero trace reserves,
and the existing independent full-vocabulary comparator. Every serialized call
used its original inputs and greedy token; no new state comparator was added.

| Existing gate/recipe | Calls | Logits | Exact |
| --- | ---: | ---: | --- |
| Full40 Hello/Capital prefill + one decode | 4 | 1,986,560 | Yes |
| `greedy4` | 10 | 3,476,480 | Yes |
| `fork2` | 6 | 1,489,920 | Yes |
| `canary2` | 5 | 2,234,880 | Yes |
| `recreate1` | 4 | 993,280 | Yes |
| `cold-owner1-01` | 4 | 1,986,560 | Yes |
| `cold-owner1-02` | 4 | 1,986,560 | Yes |

**37 calls / 14,154,240 logits**, all exact, zero nonfinite values and maximum
absolute error 0. Greedy full40 IDs: `11,353,11751,13`. Existing
publication/frontier/cancel/fork/reset/recreation/cold-owner checks passed.
The original tolerances were not changed.

Maximum sampled GPU memory: **21.1044921875 GiB/owner**, below 22 GiB. Actual
ready/shutdown cache snapshots show **3,221,225,472-byte arena backing/owner**.
Sampling is not an exact/continuous peak guarantee.

## R64 boundary and CPU budget evidence

R64 backend GPU and CPU planner/order/bit/bounds/grant gates passed. The new
synthetic R64 CPU route-root floor uses the actual planner, opaque packer and
allocator grant function: 524,288 B gather + 512 B metadata; 2,560 B host
staging for 64 groups. Existing capacity admission rejects a one-byte-short
budget. This isolates route roots, **not** a complete synthetic model estimate.

The existing real-header max-rows=8/Top8 full decoded-cache budget gate passed
for both ranks: complete device grant + owner margin **22,575,706,624 B**, below
22 GiB by **1,046,613,504 B**. It also rejects an undersized capacity.

**No independent real model R64 partial fixture exists here.** Model small
partials remain explicitly synthetic; full40/recipes cover genuine M5/Top8
R40. No R64 model numeric acceptance, invented partial golden or fresh oracle
export is claimed.

## Default host-control / NCCL audit and clippy follow-up

`MixedRankCollective::prepare` sets the private `device_participation=false`;
`HybridBf16Collective::data_location` therefore selects Host. The exact CPU
`default_host_preserves_vec_and_exact_route_schedule` gate passed. An initial
wrong CPU substring selected zero tests; that receipt is retained and is **not**
counted as validation. The subsequent qualified selector ran exactly one test.

The native backend may compile internal NCCL bindings against the pinned
2.27.5 header without enabling them. Actual header SHA256:
`4b240a1fa79858ca8cdffcf5c2fc40aeaca8b3eaec61728213715f0902aba3a8`.
No NCCL library was mapped in the sampled child process diagnostics. The
explicit library loader and communicator initialization are not selected by
the default host path. Private device/host allocation bounds remain **Unknown**;
`TpResourceEstimate::hard_cap_bytes` rejects an unbounded hard-cap claim. The
corresponding CPU contract gate passed; no NCCL provider was initialized.

CUDA check passed. Backend strict clippy (`-D warnings`) passed. Current
model/runtime clippy and model `--lib --tests` clippy passed **with one warning**:
`mixed_channel.rs:149` reports intentionally unused private `permit_device`,
`check_device_permit`, `device_enqueued`, `complete_device_payloads` methods.
Model strict clippy is consequently **blocked**, not passed. Their owner must
choose an explicit disabled-interface lint policy; the validator did not alter
production or enable a device path to silence the warning. Other targeted CPU
tests, final paper-policy tests, changed-file rustfmt and whitespace checks
passed. No production default-public-operation numerical failure was observed.

## Artifacts and release

| Artifact | SHA256 |
| --- | --- |
| Backend route library tests | `2636ee8fb65102fb69421e5926655d68e98ec34ada23462ac19448a9a2b36308` |
| Final model counter/custody tests | `bd9ee9bbf40042b397efd1cbe38338eba7d7866ea4725a5510c042463a55986f` |
| Composition E2E tests | `109446e086e473b80a99f0f73a6f962b590c6ad6af330f1bf1dc9aed9fb7afda` |
| Preserved before-fused release | `f6a65e1b6f09c0a8385fcfb0ee0b1cab5f02d2b09a341a6cd4d9e7aa81ed06f4` |

The old f6a artifact, prior 14M-exact receipts and supplied 6.514 tok/s baseline
were preserved. **Incremental performance is unmeasured.** Fresh production
release build and simple release E2E belong to the next builder/owner stage.
No public serving enablement, internal tensor/state equality, or NCCL quality /
hard-cap acceptance is inferred from these tests.

Final maintained supervisor receipts prove drained children and kernel ECHILD.
Recorded owned birth identities are absent. GPU0/1 UUID locks were reacquired
nonblocking and released; all eight GPUs were idle with zero memory/use and no
compute apps. GPU ownership is released; no foreign process was signalled.
