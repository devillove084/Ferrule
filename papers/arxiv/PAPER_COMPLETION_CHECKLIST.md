# Ferrule manuscript completion checklist

**Ferrule: Bounded Storage-Backed Weight Residency**
Lu Huanbing · Independent Researcher · `luhuanbing084@gmail.com`

Checked items describe available evidence, not universal system guarantees or
submission acceptance. Use the [artifact guide](ARTIFACT_GUIDE.md) for exact
inputs, commands and identities, and the [evidence ledger](EVIDENCE_LEDGER.md)
for claim boundaries. Do not copy engineering provenance into the manuscript.

## Scientific evidence

### Quality and controlled budgets

- [x] Separate the nine serving-source factory gates from the nine earlier-source
  gates; neither source cut enlarges the denominator of the other. Test binaries
  and the actual serving executable have distinct identities.
- [x] Complete the five-pair serving study with frozen quality checks, common
  controls, equal logical work and recorded closure.
- [x] Complete three cache-pressure points under coupled entry/byte limits.
  Preserve actual occupancy, logical charges and sampled GPU/PSS as different
  quantities; fewer evictions did not produce fewer reads or observed cache hits.
- [ ] Establish a physical capacity frontier, isolated cache-policy/byte-budget
  sensitivity or compact-weight residency impossibility, only if claimed.
  Configured budgets and sparse memory samples do not establish these results.

### Demand, waiting and overlap

- [x] Reconcile paired invocation/request/owner counters and shared-client I/O.
  Missing startup/shutdown PID-I/O and physical NAS attribution remain missing.
- [x] Use paired blocks, not individual requests, as the resampling units.
  Preserve the prespecified descriptive estimator and all five pairs; no
  coverage-guaranteed confidence interval, power or significance claim follows.
- [x] Report the separately profiled scoped-overlap observations with actual
  kernel correlations, read envelopes and userspace pread wall intervals.
  Retain full-trace rejection, wrapper warnings and unknown dropped-event count.
- [ ] Establish physical NAS-service overlap, a causal critical-path decomposition,
  instrumentation overhead or ordinary-performance effects from profiler timings.
- [ ] Add independent study replication, long/concurrent workloads and fair
  external controls before claiming broader performance or a ranking.

### Custody, faults and cancellation

- [x] Distinguish known closed/retained normal owners from physical release.
  Process drain, authenticated ECHILD and idle are not device-release certificates.
- [x] Observe two real Mixed component selectors / four cells, with clean-abort
  checks and sticky Unknown retention in both schedules. The synthetic Unknown
  trigger discards proof after real synchronization, not a hardware failure.
- [x] Observe normal HTTP cancellation followed by an exact fresh-request recovery.
  Keep the cancelled request a controlled censor, not a completed QA success.
- [ ] Prove the intended credit-conservation invariant for an integrated deployed
  path. Component host/cache accounting is not a runtime five-bucket broker ledger.
- [ ] Close the full integrated fault matrix: stale/delayed completions, cancellation
  at each stage, primary-plus-cleanup failure, owner loss, lost ACKs, original versus
  repeated submissions, retained credits, and publication/reuse boundaries.
- [ ] Establish post-Unknown physical release, general recovery or same-request
  resume. Do not infer these from process exit or fresh-session success.

## Artifact completeness

- [x] Provide one [result/input/check/figure crosswalk](ARTIFACT_GUIDE.md#result--input--check--figure)
  with scientific labels instead of run-directory names as the primary index.
- [x] Verify public earlier-qualification, paired-results, pressure, overlap and
  mixed-fault archives with their standalone checkers and tests:
  37 / 29 / 14 / 33 / 27 tests pass on Python 3.10.9.
- [x] Verify the five HTTP receipt tests on Python 3.12.4, including an export of
  tracked sources without private evidence. Document shared-helper dependencies
  and the `tomllib` requirement; Python 3.10 does not run this import chain.
- [x] Separate original data hashes, public projection hashes, immutable serving
  source and later collector/test/documentation revisions. Do not identify the
  current checkout with the measured CLI.
- [x] Document private environment/authentication, large profiler files, external
  executable/model/oracle/native payloads and the limits of public verification.
- [ ] Publish and verify a complete portable archive of the current nine-gate raw
  qualification. The public build/quality projection is not that archive.
- [x] Verify the complete portable mixed-fault archive, companion figure hashes
  and all 27 standard-library tests in a detached public-only copy. Keep actual
  snapshots/boundary observations separate from source-and-PASS-backed assertions,
  and retain external binary/source-tar requirements for native re-execution.
- [x] Provide and validate the [public paired renderer](ARTIFACT_GUIDE.md#paired-public-figures):
  all three figures reconstruct from the unmodified 623-file archive, with no
  private 840-input dependency. Twelve public-render tests verify detached byte
  identity and tamper rejection; 29 archive and 16 legacy plot tests also pass.
  Byte identity is scoped to current public provenance and full renderer/font
  pins, not arbitrary historical PDFs or unpinned installations.
- [x] Publish the [post-hoc platform inventory](ARTIFACT_GUIDE.md#post-hoc-platform-inventory)
  and pass its 29 offline tests. Preserve later acquisition time, 300 same-boot
  phase snapshots across ten paired-study arms, and unknown experiment-time
  limits. Do not infer installed RAM, physical core counts or cross-study binding.
- [x] Distinguish standard-library receipt verification, pinned public plotting
  and external-input native/GPU reproduction. Document the actual sample counts
  and post-collection wait rather than an exact two-second polling cadence.
- [ ] Recover and verify external build/model/oracle payloads if executable or GPU
  re-execution is requested. Offline receipt checks do not supply those payloads.

## Manuscript and submission acceptance

- [x] Organize the manuscript around questions, mechanisms and findings, not
  artifact construction chronology. Keep rendered prose free of project paths;
  engineering provenance belongs in the artifact guide.
- [ ] Check that all result tables/captions use the guide’s scientific labels and
  actual denominators; retain negative results without pooling or retrospective
  upgrades. Do not copy full metric tables into multiple documentation files.
- [ ] Audit formal statement types: intended/conditional invariants versus proved
  results, component observations versus integrated behavior, and accounting
  ceilings versus physical memory. No five-bucket measurement may be invented.
- [ ] Run the finite paper build after final source integration; inspect converged
  references, citations, warnings and boxes. Review final figures/tables/algorithms
  in color and grayscale, and confirm the tracked PDF matches the source.
- [ ] Complete primary-source citation/guarantee and third-party licensing audits.
  Positioning against related systems does not imply a fair benchmark comparison.
- [ ] Obtain author-confirmed declarations, funding, acknowledgments, conflicts
  and submission approval. Do not invent an ORCID, release, DOI or data deposit.

Historical exclusions and supplementary results are indexed in the
[artifact guide](ARTIFACT_GUIDE.md#historical-exclusions-and-provenance), not
reclassified as new evidence. New experiments require fresh plans, destinations
and owner authorization; this checklist is not permission to launch them.
