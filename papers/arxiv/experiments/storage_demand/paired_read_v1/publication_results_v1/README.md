# Portable paired HTTP read-schedule results

This is the **2026-10-15 completed `mapfixed-20261015-01` study**, not either earlier
stopped attempt and not the initial preregistration's older production source.
The current immutable CLI is built from `86c58a017d7c4b53490f536a3b1a133f2f7848cc`.
Both arms use that same binary; only `--expert-read-schedule` changes.

Read [RESULTS.md](RESULTS.md) for the computed public results and [REDACTION.md](REDACTION.md)
for the privacy/provenance contract. This directory is self-contained for **offline
receipt analysis**, not self-contained for rebuilding/rerunning a GPU experiment.

## Offline reproduction

Python **3.10 or later**, standard library only. No packages, imports from Ferrule,
network, model access, GPU queries, or server execution are needed. From repository root:

```sh
python3 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/publication_results_v1/check.py
python3 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/publication_results_v1/test_publication.py
```

Alternatively, copy **only this entire directory** to another location and run:

```sh
python3 -S -B check.py
python3 -S -B test_publication.py
python3 -S -B check.py --json
```

`--json` prints the complete recomputation to stdout without rewriting any receipts.
The checker rejects missing/unlisted/changed members, including separate compressed
and uncompressed hashes. It reads every arm, not only an apparent winner, and:

1. Rebuilds the fixed order and all 120 request/50 measured-QA denominators.
2. Reconciles client reports, SSE text/DONE/usage, exact worker-enqueue token IDs,
   emission ordinals, worker terminal records, and HTTP completion IDs.
3. Reconstructs all 1,020 invocation logical counters and pipeline deltas, checks
   host budgets and resets, and reconciles every nonpeak owner total and current
   charge against the shutdown ledger. Peak counters are cumulative, not deltas.
4. Recomputes 150 NFS-client deltas and 120 PID-I/O deltas from 240 request snapshots,
   checks recorded PID/birth/parent and acquisition-window continuity, and validates
   20 required-library 12/12 hash projections.
5. Checks recorded closure, retained residency, authenticated-drain outcome, idle
   samples and lock reprobes without claiming replay of private authentication.
6. Recomputes the entire prespecified five-pair analysis and all 3,125 ordered
   block-bootstrap draws, matching the independently sealed post-study audit.
7. Keeps both stopped studies separate, including their failed lifecycle journals,
   unissued slots and null aggregates. No complete-case or winner-only selection.

The tests mutate actual copied evidence: missing pair, duplicate arm, failure/null
aggregate, winner-only subset, wrong ordinals, oracle mismatch even with matching
log/trace copies, usage, DONE, HTTP IDs, counter resets, missing/null counters,
PID birth, library coverage, physical-release overclaim, stale compressed/raw hashes,
nested credential contexts, traversal and isolated-directory operation.

## Contents

| File/directory | Role |
|---|---|
| `design/` | Frozen protocol/order/request/oracle/config projections; actual source/build/toolchain pins |
| `arms/P*-*/` | Per-arm journal, small raw copies, client reports, request observations, invocation/token log events, counters, maps projection |
| `study.json.gz` | Explicit projection of the sealed final result, including the original recorded analysis |
| `poststudy-audit.json.gz` | Explicit projection of the independent post-study audit |
| `recomputed.json.gz` | Full offline reconstruction, compared byte-semantically after recomputation |
| `bootstrap-enumeration.json.gz` | All 3,125 ordered block draws, not random subsampling |
| `historical-failures.json` | Both historical failures and all their separate denominator rows |
| `final-cleanup.json` | Byte-identical small final cleanup receipt |
| `provenance.json.gz` | Original raw-tree inventories, input hashes, original global hash-pin sets |
| `manifest.json` | Exact public membership, compressed/uncompressed hashes and byte counts |
| `private-scan-receipt.json` | Owner-side recursive original-private-literal scan result |
| `validation.json`, `validation-tests.txt` | CPU validation receipts |
| `prespecified.py` | Original analysis implementation with only the dependency import replaced by local equivalents |
| `common.py`, `reconcile.py`, `check.py` | Portable validation and recomputation |
| `curate.py`, `curate_bundle.py`, `audit_sources.py`, `seal.py` | Source-specific curation/audit tools; not needed to consume the bundle |

Large JSON and selected structured logs use deterministic gzip (`mtime=0`, level 9).
This is not a raw directory tarball. No ELF, `.so`, native archive, source archive,
weight, raw process map, complete environment, private authentication receipt,
mount/export identity or network hostname is included.

## Exact immutable inputs

| Original receipt, relative to `paired_read_v1/` | Original SHA-256 |
|---|---|
| `artifacts-v3-mapfixed-20261015-01/paired-final.json` | `33459c536d5a736bb4843e80e2cf3e79016fae054e5e5664cfe859cfb089e357` |
| `plan-mapfixed-20261015-01/poststudy-audit.json` | `30d123fac5be3df3eae9a0241f75897ea24a2d30cec254c41456baed85dd17c5` |
| `artifacts-v3-mapfixed-20261015-01/final-cleanup.json` | `0f1f42d854af3e27eba09a1eb2df0dbd7ae19860015de589910b3ae556254563` |

The originals remain private, unchanged and outside this bundle. Projection hashes
are **not** original receipt hashes. An offline reader can verify this public copy
and all internal arithmetic; they cannot establish omitted original authenticity
or independently recapture a past process's maps/environment. Hashes are integrity
pointers, not signatures or evidence of raw-data public availability.

## Configuration and build/rerun boundary

See [REPRODUCE.md](REPRODUCE.md). The build commands and immutable artifact hashes
are recorded, but compiler/CUDA/CUTLASS/NVTX inputs, executable artifacts, the model,
and authenticated owner infrastructure must be supplied separately. No promise of
bitwise rebuild, future runtime equality, cold storage, or NAS backend data is made.
This bundle does not authorize another workload or permit resuming either stopped
study. Do not run the historical start/next commands against existing directories.

## Scientific/resource limitations

The primary estimand uses each arm's median of five measured QA E2E times and the
median of five paired log ratios. TTFT is first nonempty client SSE content. Goodput
is quality-qualified **serial request-window** tokens/second, not concurrent or
startup-inclusive throughput. Bootstrap intervals are descriptive n=5, not
coverage-guaranteed confidence intervals or significance tests.

Fresh servers do not imply independent/cold page cache or NFS samples. Shared NFS
client counters are not request-exclusive storage attribution or NAS disk/network
traffic. Host wait intervals and the last ticket receipt are not CUDA-overlap
proof or a complete per-ticket journal. Sampled device memory is not exact
owned-process GPU accounting; 64 MiB is the logical read reservation, not total RSS.

Startup/shutdown PID-I/O snapshots were **not collected**. Missing values stay
null/unobserved; they are not reconstructed as zero from process exit. The 240
request-boundary PID snapshots and 30 lifecycle-phase NFS deltas are present.
All owners are recorded known closed/retained; zero final logical read charges,
ECHILD and idle do **not** prove post-drop physical release. No scientific
acceptance, SOTA claim, general workload scaling or performance significance is asserted.
