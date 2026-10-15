# Private runtime-map evidence and one Ready-only calibration

This append-only adapter fixes the final-layout V3 worker without editing the
historically pinned lifecycle sources, preregistration, old bindings or failed
V3 artifacts. The historical `execution_v2.library_maps` predicate is retained
only as historical code; `final_adapter_v3/run.py` installs this adapter for the
actual worker. Old execution bindings are stale and must not be reused/resealed
in place. A fresh parent plan and new private binding are prerequisites to any
future paired execution. No ten-arm study or profiling is authorized here.

## Evidence and validation

The owned capture parent and actual server PID/birth/parent are authenticated,
not guessed from a process name, shell wrapper or current child list. Each map
observation saves exact `/proc/PID/maps` bytes to a new private directory before
ELF/library validation. `snapshot.json` binds the bytes, timestamps, PID/birth,
parent identity and executable SHA; `validation.json` preserves exact missing
required paths, deleted libraries and errors. Acquisition failure is explicitly
unavailable, not an empty successful observation. A recycled PID is never read.
No raw environment or authentication material is included in map receipts.
Directories are 0700; maps and receipts are 0600, exclusive-created and fsynced.

Proc fields are split into five fixed fields plus the complete pathname (spaces
preserved). The deletion suffix is separated from the pathname. Non-executable,
non-library deleted `/dev/zero`, memfd, shared-memory and SYSV mappings do not
imply that a pinned library is deleted. Deleted required libraries, unbound
libraries, unknown deleted files/executable mappings, inode substitution,
non-overlay device mismatches, hash mismatches and malformed observations remain
failures. Overlayfs can expose the overlay superblock device in pathname `stat`
and the backing device in proc maps. This exception requires the actual
longest-prefix mount to be overlay and its device to match pathname `stat`;
inode and all required pathname SHA checks still apply. Each discrepancy is
reported with backing-file identity explicitly unobserved, not silently ignored.

Every pin in the explicit executable build role and its recursive ELF
`DT_NEEDED`/interpreter closure is required, including all CUDA dependencies.
Other-role pins can be absent only with an explicit classification; if mapped,
they still need the same identity/hash checks. No pin is discarded. Zero bound
ELF observations produce a null required-pin result, never PASS. SHA256 covers
the current pathname file, with inode checks and device comparison where
comparable. Overlay backing-file identity, live mapped memory hashes and
physical CUDA resource release remain unobserved.

All nine same-cut factory quality gates and actual mapped CUDA pin evidence are
still reviewed by the existing final-layout reviewer. The 1,484-member immutable
source archive, evidence manifest and exact three generated noncompiled pyc
manifest dispositions are unchanged; there is no extension-based exemption.

## Single authorized calibration

The `-01` directory preserves CPU-only preparation before tightening the shared
45-second shutdown deadline. It has no launch-attempt marker, GPU query or
server launch; its receipts are not rewritten or reused.

`calibrate.py prepare` performs CPU-only review and reserves one new private
`calibration-map-ready-20261015-02/` directory. `run --sole-gpu-owner-granted`
consumes that directory exactly once, including on failed preflight. It uses the
same immutable CLI SHA `7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01`,
literal V3 Blocking argv, cohort 1/0, host 64 MiB, metadata 8 MiB and stack 2 MiB.
It uses maintained all-eight UUID locks, authenticated owned subreaper/capture,
ECHILD, absent-owner checks and stable-idle-v2. Bounds: startup 120 seconds,
total lifecycle 240 seconds, shutdown 45 seconds. No retry, cache flush, foreign
signal, conditioning workload, model inference, generative HTTP or profiling.

Readiness uses the maintained non-generative **GET `/v1/models`**, not `/health`;
its model-list availability checks are journaled without response bodies or
authorization headers. The sole server is stopped immediately after raw Ready
maps are saved. Both the literal original predicate and fixed validator are run
**offline after shutdown against the same saved snapshot**. Calibration is not
paired data, a replacement arm, CLI numerical-quality proof or authorization to
resume the permanently stopped V3 study. The absent original failing raw maps
cannot be reconstructed; reproduced calibration cause must not be presented as
a direct observation of that lost historical snapshot.

Use the repository's exact Python 3.12 with `-S -B`. CPU tests:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/tests -p test_library_maps_v1.py -q
```

## Recorded 2026-10-15 result (not study data)

Exactly one server was launched in `calibration-map-ready-20261015-02/`. It reached
Ready in **13.767396539 s** with zero inference requests. There were 137 maintained
GET `/v1/models` availability checks (one HTTP 200), no generative HTTP. The raw
snapshot SHA256 is
`98a68c65098c37004c355a3841b5e543b7db39806a51f728b2a24fc73ddeba40`.
The only deleted pathname is `/dev/zero`; all 12 required library paths exist in
the snapshot and their pins match. The original predicate fails on that snapshot.

The first fixed assessment was also failed: its overly strict device equality
check rejected seven overlay-backed system libraries (`maps` device `fd:11`,
pathname `stat` device `0:30`, matching inode). That `FINAL.json`, original offline
comparison and first validator source are preserved, not relabeled PASS. The
corrected overlay-aware validator, tested **offline on the same snapshot without
another launch**, passes **12/12 required pins**, zero optional pins, zero missing
or deleted libraries; seven backing-file identities remain unobserved. A separate
`offline-comparison-overlay-v2.json` records the new assessment and validator SHA.
This is a reproduced cause of the old predicate's failure, not recovery of the
missing raw maps from the earlier stopped V3 arm.

Owned server PID/birth was **124405 / 133060401**; capture PID/birth was
**124404 / 133060398**, distinct from authenticated worker 124403. Server shutdown
returned 0 in **0.815416 s**. Maintained supervisor ECHILD/proven drain, absent
recorded owners, stable-idle-v2 accepted samples 0/1/2, all-eight lock release and
nonblocking reacquisition were verified. The complete lifecycle took **21.154806 s**.
Physical resource release and live mapped-memory hashes are still unobserved.

CPU validation: 135 helper tests (including 19 map/calibration tests), 16 final
adapter tests, and 15 V3 runner tests passed. The broader historical paired suite
retains four failures involving stale production-source/V2 seal assumptions;
production code and historical seals were not changed to hide them. The current
nine-gate reviewer still passes; all archived/mapped CUDA pins and immutable
source/three-pyc dispositions remain intact.

Next execution needs a **fresh parent plan and authorization**, new private
binding/review/execution destinations with the new adapter pins, and independent
lock/idle admission. Neither the permanently stopped V3 study nor this consumed
calibration may be resumed. No ten-arm study or profiling was launched.

Only source, tests and safe summaries may be staged. Private bindings, map bytes,
worker authentication receipts and calibration raw artifacts stay untracked.
