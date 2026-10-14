# Curated paired-study publication v1

This is a small **CPU-only publication/backup stage**, not a resumed experiment,
new GPU qualification, or V3 candidate acceptance. No Git/model/GPU/server
commands are run by the curator or checker. All inputs are ordinary files within
`paired_read_v1`. Existing V1/V2 files and the eight V3 files stay unchanged.

## Exact staging whitelist

`STAGE_FILES.json` is the authoritative literal file list. Its paths are relative
to `papers/arxiv/experiments/storage_demand/paired_read_v1/`, portable, UTF-8 text,
regular and nonsymlinked (including parent components). It groups files for main's
separate backup/commit stages; the checker rejects duplicates, traversal, binary
payloads, unauthorized environments/credential fields, hash changes and a stage
of 10 MiB or more. **Stage only listed paths—never the whole directory or a raw
manifest's dependency inventory.** Main alone coordinates Git staging/commit/push.

### Initial preregistration V1 — source, tests and safe metadata

`README.md`, `__init__.py`, `analysis.py`, `build_adapter.py`, `cases.json`,
`cpu-sealed.json`, `cpu-tests.log`, `cpu-validation.json`, `oracles.json`,
`order.json`, `profile_audit.py`, `protocol.json`, `protocol.py`, `quality.py`,
`requests.json`, `run_paired.py`, `source-cut.json`, `stage-manifest.json`,
`test_paired_read_v1.py`, `test_quality_v1.py`.

Source pin metadata is not a source archive, model/weight copy or executable.
Frozen public-canary token IDs are not authentication credentials. Case
`environment` objects contain only the reviewed public controls/library-path
settings, never inherited arbitrary process environments.

### V2 reviewed binding and failed-study backup

`EXECUTION_V2.md`, `RESULT_V2.md`, `actual-binding-v2.json`,
`actual-build-review-v2.json`, `actual-stage-manifest-v2.json`, `cpu-tests-v2.log`,
`cpu-validation-v2.json`, `current_binding_v2.py`, `execution-sealed-v2.json`,
`run_actual_v2.py`, `test_actual_v2.py`, `test_stopped_closeout_v2.py`.

The binding/review/manifests are safe **hash/provenance metadata**, not their
referenced raw files or libraries. Their external artifact paths/hashes do not
allow staging/copying those dependencies. No actual CLI/native binary, source
archive, weights, runtime map or external qualification artifact is included.

### Prospective V3 — exactly the eight new unbound files

`plan_v3.py`, `protocol-v3.json`, `cases-v3.json`, `PLAN_V3.md`, `test_plan_v3.py`,
`cpu-sealed-v3.json`, `v2-configuration-rejection-v3.json`, `cpu-validation-v3.json`.

V3 has zero lifecycle attempts and null candidate binary/source cut. Explicit
cohort1/deferrals0 flags do not fix the TP1 accessor bug. A future **independent**
study requires main's production fix, a newly compiled immutable artifact,
truthful versioned semantic build review, fresh same-cut constructor and both-arm
full40/fresh/ordinary/cold qualification, new V3-bound execution receipts and
separate owner authorization. It never resumes or replaces the stopped V2 study.

### Curated publication files

`publication_v1/README.md`, `publication_v1/check.py`, `publication_v1/curate.py`,
`publication_v1/startup.log`, `publication_v1/study.json`,
`publication_v1/supervision.json`, `publication_v1/closure.json`,
`publication_v1/nfs-client.json`, `publication_v1/custody-idle.json`,
`publication_v1/bindings.json`, `publication_v1/STAGE_FILES.json`,
`publication_v1/manifest.json`, `publication_v1/test_publication.py`,
`publication_v1/stage.py`, `publication_v1/STAGE_FILES.tsv`.

No compression is needed; the startup log and projections are small text files.

## Raw-copy / projection policy

- `startup.log` is the **only verbatim raw text copy**: the small first-attempt
  startup failure log, audited for credential assignments and private paths.
  Its copy SHA256 equals the original log SHA256.
- Every curated JSON uses `redacted-publication-projection.v1`, declares
  `redacted_projection=true` and `replaces_original_receipt=false`, names its
  omitted fields, and records each original path and **original** SHA256.
- `manifest.json` separately records the **published byte** SHA256 and size of
  each staged file. These hashes are not mislabeled as hashes of a substituted
  raw receipt. The unsigned manifest excludes its own self-hash; the checker
  verifies its schema, exact whitelist coverage, size arithmetic and scientific
  invariants. This is integrity checking, not a signed or new authenticated proof.
- `study.json` retains all five paired blocks, ten lifecycle slots, 120 unissued
  requests and 50 scheduled/zero-eligible measured QA. One lifecycle was attempted,
  none completed; no HTTP token IDs or timing/goodput/read-worker statistics exist.
  Unobserved outcomes remain null, not zero-performance claims.
- `supervision.json` retains recorded return/deadline/drain/ECHILD and PID/birth
  relations/signals but omits raw stdout/stderr, handshake material, launch IDs
  and boot identity. Offline checking **does not replay original authentication**.
- `closure.json` preserves failed startup/capture status and missing owner summary.
  Known process exit plus idle does not repair the **unknown** owner/read resource
  ledger or establish universal physical release.
- `nfs-client.json` retains the actual three sampled acquisition windows and
  selected client counters (prelaunch, shutdown-pre/post). Export/server/mount
  paths, raw mount text and reader/namespace identities are omitted; a labelled
  opaque identity digest preserves equality checks. Shutdown differences are
  checked exactly, with no negative/reset deltas clamped to zero. There are no
  Ready/startup or HTTP windows. Shared NFS client deltas remain non-attributable
  to this server; physical NAS disk/network bytes remain unknown.
- `custody-idle.json` retains recorded post-ECHILD idle samples with numeric GPU
  indices/memory/utilization and empty-app counts, but redacts physical UUIDs and
  raw lock FDs/inodes. All-eight local lock reacquisition/release and three zero
  samples do not constitute a CUDA release certificate or global reservation.
- `bindings.json` retains V1/V2/V3 protocol/case/receipt/artifact hashes and the
  old CLI serving failure/unbound V3 status, without embedding binary/archive/map
  bytes or huge external dependency inventories.

All originals under `admission-20261015-01/` and
`artifacts-current-20261015-01/` remain private, append-only and **excluded from
staging**. In particular, never stage raw `worker-launch.json`, `launch.json`,
`registration.json`, `assessment.json`, `lifecycle.json`, `supervisor.json`,
`paired-final.json`, observer maps/environments, raw mountstats/export text or
lock/handshake receipts. They contain local ephemeral launch credentials or
private host/export detail even if a corresponding projection is published.
The manifest records hashes/sizes of all 76 prepublication originals solely to
check that they were not changed. It does not authorize raw-copy publication.

## Offline checks

From repository root:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/publication_v1/check.py
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/publication_v1/check.py --publication-only
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/paired_read_v1 -p 'test_*.py' -v
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/plan_v3.py check
```

The default checker verifies whitelist files and all preserved original hashes in
this workspace. `--publication-only` needs only this copied publication directory:
no private raw files, target artifacts, runner imports, CUDA libraries, model or
network are consulted. It checks published hashes and conservative recorded
scientific/custody/NFS invariants, not full original receipt authentication.

`curate.py` is an exclusive one-time build for this exact stopped study. Do not
rerun over existing publication outputs. Changes require a new publication
version, never rewriting frozen receipts or silently relaxing failure criteria.


## Separate explicit V3 integration

The eight original prospective V3 files above are unchanged and remain unbound.
The separate `v3_execution_integration` stage group adds `current_binding_v3.py`,
`run_v3.py`, `test_run_v3.py`, `EXECUTION_V3.md` and its truthful CPU validation
receipt. See `EXECUTION_V3.md` for exact commands and the real corrected-build
identity. No future candidate PASS, binding, execution seal or GPU result is
included. All 76 prepublication original hashes are still enforced. The
publication's semantic projections remain the failed V2 study and the original
unbound V3 declaration, not a new execution or replacement of history.

`STAGE_FILES.json` is the complete authoritative list; `STAGE_FILES.tsv` contains
the same groups with exact repository-relative paths. The stdout-only generator
requires an explicit CLI format and verifies all whitelist files first:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/publication_v1/stage.py --format json
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/publication_v1/stage.py --format tsv
```

Neither command stages anything or calls Git. Do not use historical whole-folder
stage suggestions embedded in old frozen metadata: only the current literal
whitelist is authorized. Existing 10 MiB text-only bounds are stricter than the
20 MiB maximum; the manifest itself is included in the checked total.
