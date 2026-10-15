# Main-owner handoff — actual fault publication, 2026-10-15

**Complete and ready for review; not staged or committed. No new models.**
The paper author may now reference this existing publication. Manuscript edits,
placement, Git staging, commits and push remain with the main owner.

## Exact ownership / staging boundary

Only these new deliverables were written:

- `papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_fault_v1/`
- `papers/arxiv/figures/fault-custody-contract.py`
- `papers/arxiv/figures/fault-custody-contract.pdf`
- `papers/arxiv/figures/fault-custody-contract.png`
- `papers/arxiv/figures/fault-custody-caption.md`

`STAGE_PATHS.txt` is the complete repository-relative **36-file allowlist**, one
path per line, including this handoff and the allowlist itself. Use that exact
list for review/staging; do not indiscriminately force-add the whole ignored
publication tree. `.cache/`, `__pycache__/`, `.preview/`, and `.plot-env/` are
excluded by the local `.gitignore` and by the list. No package/environment files,
private inputs, original frozen tree, binary, archive, shared library or model
are in the allowlist. Combined publication + figure products are about **1.5 MiB**,
well under 10 MiB. No parent README/artifact docs, manuscript/main/sections,
production source, previous raw data or Git state was altered.

## Scientific summary for the paper owner

- Actual component execution: **2 selectors × 2 schedules = 4/4 cells PASS**,
  zero selector retries. Tiny width-16 GDN/GQA, four experts/top-3, BF16 history/KV
  and F32 SSM; not a 35B/CLI or integrated broker E5 result.
- Tx1 commits frontier 1/slot 0; tx2 errors after real routed-down submit and
  successful actual consumer completion/upload wait. Clean retains owner proof,
  returns 512 B working state/two transient KV slots, verifies committed
  KV/convolution/SSM bit equality, and executes then aborts healthy in-test tx3.
  Post-abort state is 1,024 B, not zero; cache is 61,440 free + 4,096 charged.
- Synthetic Unknown **discards the successful owner proof**. Two later independent
  compute and two upload synchronizations cannot discharge it. Typed aborts stay
  Unknown; reset/admission/KV release/publish/shutdown reject. Retained state is
  1,536 B, one active transaction; cache 58,618 free + 6,918 charged. The 6,918 B
  Unknown value is a **subset**, not another capacity bucket (774 pending +
  6,144 scratch). The separate 318,720 B workspace overlaps owner accounting and
  is not added into a physical total.
- **Post-Unknown KV / SSM contents are unobserved.** Only committed metadata is
  unchanged. Independent fence outcomes/rejected methods are explicitly
  **source + actions + PASS backed**, not fabricated individual API-error or
  fence receipts. Only execution/abort use typed Unknown assertions; the other
  rejected APIs assert errors without stronger downcast claims.
- All four host readers close independently with zero current/Unknown charge.
  Lookahead wastes one 774 B read per cell. This does not release GPU quarantine.
- Only clean's separate explicit owner drop + allocator trim observes fixture
  requested bytes 0 → 0. NVML idle, authenticated ECHILD and local lock release
  are not device-wide CUDA release certificates. The public custody projection
  retains 26 NVML samples, including both utilization-only unaccepted transients.

## Validation actually run

1. `python3 -S -B .../publication_fault_v1/check.py --figures papers/arxiv/figures`
   — **PASS**. 19 pinned payload files, 839,845 bytes; two actual 40-row reports,
   corresponding stdout/stderr, nine pinned source references, full 1,487-file
   source hash inventory, custody/history and independently derived projection.
   Figure source/PDF/PNG hashes and dimensions checked without third-party imports.
2. `python3 -S -B -m unittest discover -s .../publication_fault_v1 -p 'test_*.py' -v`
   — **27 tests PASS** (22 evidence/corruption/portability + 5 figure tests).
   Generated corrupt metric/source/frontier/fault-count/false-quiescence/
   false-content/refunded-custody inputs rejected. Rehashed false projections
   also rejected, not merely detected by old file hashes.
3. **Isolated copy** to a fresh temporary directory: all public files plus only
   the four companion files; no `target/`, private source, model or repository
   checkout. `-S -B` checker and all 27 tests passed with a minimal environment.
   The existing plotting interpreter reproduced byte-identical PDF/PNG and
   figure metadata there via `fault-custody-contract.py --check`.
4. Optional private **read-only** source audit passed: 1,487 frozen files and
   1,487 tar members; 158 original plan/manifest pins; 33 actual-handoff artifact
   pins; 13,268 external dependency pins; 111 compiled dependency pins;
   four native archives and 12 runtime libraries. No frozen-source writes,
   GPU queries, GPU launches, Git commands, or original report modifications.
5. Repeated curation produced byte-identical manifest/source/custody/history/
   projection metadata. No original measurement was synthesized or corrected.
6. Renderer: preinstalled Python 3.12.4 / Matplotlib 3.8.4 / NumPy 1.26.4 /
   Pillow 10.3.0 / FreeType 2.12.1 / zlib 1.2.13, Agg, pinned DejaVu Sans fonts.
   No package installation or new environment. PDF dates omitted, PNG metadata
   fixed. Preview inspected; blue dotted baseline, teal solid clean, brick dashed
   Unknown, shape/hatch distinctions, no text collisions or clipped labels.
7. Python AST parsing and scoped publication scan passed: no private home paths,
   GPU UUIDs, private endpoints or credential-valued assignments; no binaries,
   shared objects, tar/model files or symlinks. This is an allowlisted projection
   plus scoped scan, not a universal secret-detection certification.

No Cargo build, GPU test, model run, profiler, NVML probe, package install, Git
stage/commit/branch/push, or manuscript build was needed or performed here.
Data/check/test validation used Python 3.10.9 and isolated rendering Python 3.12.4.

## Pins

- Actual final handoff: `8f653a69707d057160118f536e77aa061209eb0d3914222223d7228d2be5abda`.
- Component source: `79d684f5c79fc6c3961161111ae6efd1a37ea8ea`.
- Archive source: `bd0673579286e614079ccfe5d972be19f860e9e8`.
- Executable: `e531be47a5113cc53665fdc6576cf389b56c29cb7e16f275efec52b0f79b6903`.
- Public payload manifest: `5bd65d0706dda32185086f7c8eebf9533ab04856ff39a34d91dc974e6a367c71`.
- Public source manifest: `aa9d0e547a0f89a43519be3bf1fd89667b3a98e00c8d161d039833f1a44138ff`.
- `fault-custody-contract.pdf`: `c22b09c187db5c19b9f240a15c6ac3af6542b5c72b8ca8afd0b6763082dfd1d8`.
- `fault-custody-contract.png`: `6ec61f608aa9094f46b73bc30505bfd43e6fe6e0e497ae940a1384faea8f2fd2`.

Full pins, library metadata, evidence typing and source-line navigation are in
`source-manifest.json`, `figure.json`, `README.md`, and `SCHEMA.md`.
`archive.py` is only an optional curator/private pin verifier; normal checking
and rendering use the public bundle alone. Hashes are drift detectors, not
signatures or independent authentication of redacted handshakes.
