# Storage-demand TP1 paper case

Status: **CPU prepared; -01 failed before launch; -02 completed with scoped
logical-demand evidence; scientific gates remain open** (`2026-10-13`).
This is a paper-only storage-demand case, not a vLLM reproduction and not a
TP2 qualification.

## Attempt chronology and current result

The initial middle entry was planned/unexecuted; final CPU entry validation
was canceled, with no GPU workload or read-counter result at that stage.

- [`-01 report`](runs/storage-demand-tp1-middle-20261013-01/report.json):
  authorized attempt failed before server launch with a harness
  `AttributeError` for missing `subprocess.DEVNULL`. Planned=4, dispatched=0,
  not-run=4; no request, read, or quality data. This was not a model/OOM
  failure. The failed receipt remains historical evidence, not overwritten.
- [`-02 report`](runs/storage-demand-tp1-middle-20261013-02/report.json) and
  [`case`](runs/storage-demand-tp1-middle-20261013-02/case.json): one authorized
  TP1/PP1/EP1 DEBUG-instrumented run (`n=1`), planned=4, dispatched=4,
  terminal=4, success=4, failed=0, not-run=0. Hello2 passed HTTP/usage checks
  only (`reference=null`); QA, arithmetic, and Chinese passed 3/3 limited
  frozen-reference visible-text/eight-output-token-usage checks. Total usage
  output tokens=26; generated IDs were not exposed. QA was one canary, not
  the extended QA workload.
- The 30 request-period invocations recorded 69,114 logical checkpoint reads /
  36,240,064,128 bytes; 88,763 facade H2D copies / 36,240,248,688 bytes;
  expert uploads=11,519, evictions=11,455, hits=0; artifact uploads=34,557.
  Startup reads are outside these sums; uploads are not a cache-miss counter,
  and facade H2D counters are not physical bus traffic.

Physical NAS/network/block-device I/O remains unknown and page-cache state
uncontrolled, not storage-cold. DEBUG overhead is unmeasured; `n=1` provides
neither comparative performance nor steady-state/statistical acceptance.
Sampled memory is not an exact peak, hard cap, or proof of owned-GPU
attribution. Authenticated drain/`ECHILD`, wrapper post-idle, and
[independent observational idle closeout](../../../../target/validation/storage-demand-middle-gpu-20261013-02/closure.json)
do not establish final ledger/device-fence/source-completion proof:
`resource_contract_complete=unknown` and raw supervisor
`gpu_quiescence=unverified` remain unchanged. Physical storage necessity,
capacity, full numerical quality/exact IDs, and safe-blocking/async overlap
acceptance remain open.

## Frozen owner command

The authoritative argv is in [`case-middle.json`](case-middle.json). It binds
`target/validation/storage-paper-current-release-20261013-01/ferrule-storage-paper-current`
(SHA256 `36e6aaa54d1aff7c8cc35d63dcee0df1cad91fcc8f6d3e746369504f2c70a58c`)
to `/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8` with:

- `--backend cuda --engine resident --model-profile qwen35-mixed-bf16`
- TP1/PP1/EP1/thread, one owner, selected ordinal 0
- `--expert-prewarm lazy`; no pageable or pinned host expert-cache flags
- context 128, packed rows/tokens 8, active sequences 2
- device budget 7168 MiB, workspace 2048 MiB, KV 16 MiB, tensor limit 1024 MiB
- middle device cache: 64 entries / 335544320 bytes, including 8 MiB scratch
- request queue 2 and max inflight requests 2
- `CUBLAS_WORKSPACE_CONFIG=:0:0`
- `FERRULE_LOG=info,ferrule_performance=debug` and JSON logs

Small and big candidates are recorded as 8 / 67108864 and 128 /
671088640 bytes. They are not executed here.

## CPU preparation evidence

- [`cpu-help.json`](cpu-help.json): exact current binary `serve --help` passed
  GPU-hidden with the complete registered argv flag set.
- [`canaries.json`](canaries.json): three existing HF canaries (`qa`,
  `arithmetic`, `chinese`), input lengths 5/5/3 and frozen eight-token
  generated prefixes from the existing reference loader.
- [`checkpoint-metadata.json`](checkpoint-metadata.json): all 14 real NAS
  shard headers/index extents and file identities checked; 0 tensor payload
  bytes read.
- [`pins.json`](pins.json): 702 complete production/runtime/reference/workflow
  pins. The current source/handoff and 12 runtime-library hashes are retained.
- [`cpu-checks.json`](cpu-checks.json): 15 bounded contract/negative checks;
  GPU queries, locks, and runs were all zero in this preparation receipt.

Exact generated-token PASS remains blocked until the run exposes/audits actual
IDs. Text/usage equality alone is not an exact-token or full-model numerical
claim. Historical Mixed full40/fresh-quality receipts remain historical pins,
not current-binary qualification.

## Offline receipt reproduction (CPU only)

From the repository root, the existing stdlib-only analysis can be reproduced
with these exact commands (no server launch, GPU query, or checkpoint payload
read):

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/analysis/tests -v
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/analysis/summarize.py
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/analysis/summarize.py --check
```

Generation writes only `analysis/output/summary.json`, `TABLES.md`, and
`REPORT.md`; `--check` recomputes and compares their bytes without writing.
See [`analysis/README.md`](analysis/README.md) and
[`analysis/output/REPORT.md`](analysis/output/REPORT.md) for the sealed input
hashes, consumed receipt pins, deterministic tables, and counter caveats.
The frozen inputs/receipts are never rewritten or automatically resealed.
Final CPU validation (`2026-10-13`): all 8 analysis tests passed; read-only
`--check` passed with 41 input hashes, 39 consumed receipt pins, and
0 divergences. This is receipt/analysis consistency, not new scientific
qualification.

## Reuse and custody

Any further authorized coordinator must bind this TP1 manifest to the existing
`tp2_resident` custody primitives: sealed `launch_guard`, authenticated
subreaper/supervisor, `run_target`, request observer, selected-GPU UUID lock,
pre/post idle checks, and normal shutdown/unknown-descendant fail-closed
checks. This manifest must **not** be passed to the fixed TP2 dispatcher and
must not add a parallel server harness.

The -02 observation is logical, per-invocation `mixed_decoder_invocation`
JSON: checkpoint read calls/bytes/source checks, expert hit/upload/eviction and
artifact/H2D counters, retained as raw lines with request/phase anchors. These
are reader/syscall-level counters, not physical NAS/NFS IO. Optional PID
`read_bytes` and NFS mountstats remain shared/ambiguous and cannot prove server
physical disk reads. `PssAnon` and physical host cap remain null.

The requested schedule contains 4 initial requests plus QA excluded-first,
2 warmups and 5 measured requests = 12, while the request ceiling supplied
for this case is 10. The QA extension is therefore explicitly blocked rather
than silently run or silently truncated. Resolve that protocol conflict before
any authorization for an extended run.
