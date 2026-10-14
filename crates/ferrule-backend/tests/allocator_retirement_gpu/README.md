# Allocator V3: real CUDA small gate

This target compiles the **unchanged frozen** production allocator/runtime and
links real `libcuda.so.1`. It does not include either CPU mock driver/runtime.
Every intercepted driver call is forwarded to the real driver. Test-only gates
pause a real host callback/driver return; the retirement Unknown case replaces
successful query/synchronization statuses with 999 after the real calls return.
The frozen runtime's existing GraphFault hooks cover context-shutdown faults.
These are controlled return-status tests, not hardware-fault qualification.

Run one exact ignored selector in a fresh supervised process. Never run the
entire test target: it also inherits unrelated runtime tests, and intentional
Unknown cases retain resources until process exit. A real, single visible CUDA
device is mandatory. CUDA absence is a failure, not a skip.

## Fresh qualification

- Freeze manifest SHA256:
  `44a551c3772e5456a842d95133e9cbdd29dadafcc3d8fb29f15205cdb7574344`.
  All 14 frozen entries matched before preparation and after execution.
- Receipt directory:
  `papers/arxiv/experiments/tp2_resident/runs/allocator-v3-fresh-20261011-02/`.
  `CLOSEOUT.json` binds source/binary pins, raw receipts and parsed metric phases.
- Build: tao CUDA 12.4.99, sm_86, production oracle=0; 102.491 seconds,
  authenticated supervisor with a 600-second preparation limit.
- Device: physical GPU0, RTX 3090,
  `GPU-b980f22b-58a3-fc88-21c1-ae86053d8535`.
  CUDA/NVIDIA visible masks both equal that UUID; tests use logical device 0.
- CPU: 10 V3 races + 14 teardown + 29 batch + 7 allocator = **60**.
  Separate no-default-features retirement phase: **29**.
- GPU: **18/18**, each a separate exact process with a 60-second limit:
  9 new cases here, 4 frozen library context-drop cases, 5 public allocator cases.
- Missing-device regression: both formerly skipping public allocator tests
  failed explicitly with exit 101, zero passed/one failed, in CPU-only processes.
- Test-scope clippy with `-D warnings` and targeted rustfmt check passed.

## What the nine new cases prove

| Case | Evidence |
| --- | --- |
| N=8, two registered streams | Last Drop creates/records zero events; sealing creates one batch and exactly two fences, not sixteen. Pending 2048 bytes cannot be reused. Completion/poll permits original-pointer reuse without another driver allocation. |
| 256 MiB pending-byte pressure | Four 64 MiB batches remain pending; compatible free extent succeeds, fresh 80 MiB driver growth fails. Reserved 320 MiB is not treated as a forbidden working-set budget. |
| 256 pending batches | 16384 blocks form 256 batches/fences. Safe extent succeeds; incompatible driver growth fails. |
| Weak gone during owned-stream Drop | Independent ticket remains Pending while last-Arc destructor and real CUDA stream synchronization are blocked. New registration cannot prune it or reuse the block. Completion permits one remaining-stream fence and reuse. |
| Capture with Recorded proof | Capture begins between event query and publication. Pending proof survives; after capture, original two fences suffice, no fallback sync. |
| Capture with Complete proof | Completed proof survives the same publication barrier; no event re-query or replacement fences needed after capture. |
| Retirement query + fallback Unknown | Entire eight-block batch/2048 bytes and 64 MiB reservation remain retained. Zero frees, context syncs and primary releases; process exit is required. |
| Known shutdown | Pending/reserved become zero; one driver free and one primary release. Explicit shutdown plus final context Drop each perform their existing context completion check. |
| Shutdown fault | Actual context sync followed by injected 999; zero segment frees and primary releases. |

Ordinary positive retirement phases performed zero context/event synchronizations.
Explicit test stream synchronization and final-context synchronization are logged,
not hidden in a zero-sync claim. Positive final trims had zero pending/reserved
bytes. Unknown cases intentionally did not trim; authenticated process drain and
subsequent all-eight-GPU idle checks establish exit cleanup, not in-process reuse.

## Reproduction and limits

The small runner is
`papers/arxiv/experiments/tp2_resident/allocator_v3_gate.py` (paper paths are ignored
by the repository's existing ignore policy). It reuses `contract` UUID locks and
`test_vllm_qwen35.run_process`/owned-subprocess supervision; no new custody
framework or model import is introduced. Use the base Python 3.12 with `-S -B`.

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/tp2_resident/allocator_v3_gate.py prepare --id NEW_ID
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/tp2_resident/allocator_v3_gate.py cpu --id NEW_ID
# Requires a new explicit GPU ownership grant, not authorization from this README:
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/tp2_resident/allocator_v3_gate.py gpu --id NEW_ID
```

The small backend gate passes. **No full model was run or qualified.** A later
full-model run needs separate authorization and fresh numerical/performance
receipts; it cannot inherit these small-gate event counts or historical tok/s.
No foreign process was killed. Every GPU process drained; all eight GPUs were
idle afterward and the selected UUID owner lock was released.
