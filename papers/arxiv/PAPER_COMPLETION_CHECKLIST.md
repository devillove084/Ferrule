# Ferrule manuscript completion checklist

**Ferrule: Bounded Storage-Backed Weight Residency**
Huanbing Lu · `luhuanbing084@gmail.com`

Checked items describe available evidence, not universal system guarantees or
submission acceptance. Use the [artifact guide](ARTIFACT_GUIDE.md) for exact
inputs, commands and identities, and the [evidence ledger](EVIDENCE_LEDGER.md)
for claim boundaries. Do not copy engineering provenance into the manuscript.

## Remaining integration gates

Final cross-section/source integration and the finite paper build are complete:
the current tracked PDF has 41 pages, converged references/citations, and no
LaTeX/BibTeX warnings or underfull/overfull boxes. The two new plots passed the
explicit renderer check and color/grayscale visual review; rendered-page review
covered Algorithms 1–3, the experiment-family and logical-host tables, related
work/new bibliography entry, abstract and conclusion. The build-local receipt
records exact input/output hashes, tools, page coverage and review limits.
This is scoped integration acceptance, not a full manual submission proof,
licensing approval, author approval or a new runtime qualification. Those
remaining gates stay unchecked below.

Physical-frontier, integrated five-class conservation/fault-matrix, universal
physical-release, DeepSeek-V4 execution and broader significance/generalization
items below are **unproved extensions or explicit scope exclusions**, not
unstated requirements of the bounded architecture result. Keep them unchecked;
restore them as required gates only if the manuscript claims those results.

## Scientific evidence

### Quality and controlled budgets

- [x] Separate the nine serving-source factory gates from the nine earlier-source
  gates; neither source cut enlarges the denominator of the other. Test binaries
  and the actual serving executable have distinct identities.
- [x] Complete the five-pair serving study with frozen quality checks, common
  controls, equal logical work and recorded closure.
- [x] Record the separately qualified persistent-reader nine-gate GPU stage:
  one worker creation, 159/2,559 reuses at selected live endpoints, 10 MiB idle
  logical charge and zero live joins. Final-owner join remains unobserved.
- [x] Record the grouped synchronous 14-gate actual GPU quality stage as backend 1,
  model 5 and composition 8, with unchanged numerical criteria and budgets.
- [x] Separate four actual public-CLI acceptance servers / 48 correct requests
  from the completed eight-server HTTP pilot / 96 requests / 40 measured QA8 /
  320 measured tokens. Keep the two zero-start pilots permanently failed with
  unissued requests and null metrics; do not pool them or the acceptance cells.
- [ ] Qualify DeepSeek-V4 end-to-end execution or another adapter before making
  a tested model-generality claim. It is currently untested and out of scope.
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
- [x] Record grouped whole-test timing as eight completed unprofiled pairs,
  four per read schedule, with approximately 3.9% lower descriptive median
  elapsed time; preserve the containing 24-job stage's failed profile gate.
  Do not relabel whole-test time as HTTP or model-only latency.
- [x] Record qualified partial selected-process profile pairs for both schedules
  at the same frozen application under disclosed within-pair controls: native
  stream-sync rows 382,743 → 257,163 (−5/logical upload), matching H2D/kernel work.
  Keep full-trace acceptance false, drops unknown, and old failed captures failed.
- [x] Preserve the rejected retained-upload regressions (+54.51%/+56.49% whole-test
  time) and separate partial pin/free observations. Do not present overlapping
  host API sums as an exclusive causal explanation or retain dead production
  code solely for historical reproduction; use pinned Git and experiment evidence.
- [x] State the HTTP pilot's conditional policy effect and counterexample:
  lookahead TTFT/E2E −8.47%/−7.34%, phase goodput +6.52%; blocking TTFT +4.96%,
  E2E nearly flat, goodput +3.74%. Two paired blocks per schedule, with shared-
  cache independence unestablished; not independent requests or a pooled
  four-pair schedule effect, and no CI/significance claim. The frozen pilot did
  not benchmark the post-cleanup resolver. Ordinary defaults are blocking plus
  synchronous; omitted transfer policy selects grouped only for supported
  explicit bounded lookahead. Current selector validation is CPU-only, not
  post-cleanup GPU or numerical-quality requalification.
- [x] Distinguish client-first-content TTFT, client-terminal E2E and phase
  goodput including gaps from request-only service rate and the older serial-
  window goodput. Do not mix these with the older five-pair read-schedule effect.
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
- [ ] Publish and verify a complete portable archive of the earlier serving-cut
  nine-gate raw qualification. The public build/quality projection is not that archive.
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
- [x] Verify both newer curated manifests: 10 worker/transfer summary members
  and eight HTTP summary members, with matching byte counts and SHA-256.
  These checks do not rederive ignored raw reports or remeasure GPU results.
- [x] Preserve the newer measured CLI production/archive identity separately
  from later orchestration, grouped component experiments and older serving
  source. Current cleaned HEAD is not a measured or newly qualified CLI.
- [x] Record frozen CPU review receipts as historical evidence, including the
  HTTP 220-test receipt and worker/transfer control checks, without pretending
  this documentation integration reran them or they are model-speed data.
- [x] Disclose newer raw reports, profiler databases, binaries and build/model
  dependencies as local/external. Manifest hashes, sealed/no-op preflights and
  plans pending fresh GO are not a portable public GPU re-execution kit.
- [ ] Publish and independently verify a complete portable newer raw evidence
  and physical-frontier corpus if those stronger reproducibility claims are made;
  current curated summaries and the old renderer do not supply one.
- [ ] Recover and verify external build/model/oracle payloads if executable or GPU
  re-execution is requested. Offline receipt checks do not supply those payloads.

## Manuscript and submission acceptance

- [x] Organize the manuscript around questions, mechanisms and findings, not
  artifact construction chronology. Keep rendered prose free of project paths;
  engineering provenance belongs in the artifact guide.
- [x] Update the abstract, introduction, related work and conclusion to lead
  with bounded storage-backed materialization/ownership, distinguish read and
  transfer comparisons, and state conditional performance with the regression.
- [x] Verify 17 cached primary-source blobs against both Git blob IDs and SHA-256;
  cite the pinned official SGLang expert-pack source with access date 2026-10-18,
  inspect its loader/runtime/model/parallelism constraints and SPDX headers,
  and copy no upstream implementation. Full release/run qualification is not claimed.
- [x] Preserve unmatched resident TP2/vLLM observations and original controls in
  the artifact guide, not as a manuscript speedup/SOTA comparison. Maintain the
  BF16/F32 versus FP8-storage boundary and explicit DeepSeek-V4 exclusion.
- [x] Check assigned lead/artifact local links, defined citation keys and label
  references (including figure-macro labels), manuscript provenance-text exclusion,
  abstract-only changes to `main.tex`, and scoped Git whitespace. These CPU/source
  checks do not pre-complete final LaTeX, figures, PDF or author review.
- [x] Check that all result tables/captions use the guide’s scientific labels and
  actual denominators; retain negative results without pooling or retrospective
  upgrades. Do not copy full metric tables into multiple documentation files.
- [x] Audit formal statement types: intended/conditional invariants versus proved
  results, component observations versus integrated behavior, and accounting
  ceilings versus physical memory. No five-bucket measurement may be invented.
- [x] Run the finite final paper build with latexmk/pdfLaTeX/BibTeX: 41 pages,
  resolved references/citations, no warnings or underfull/overfull boxes. Bind
  the tracked PDF to recorded TeX/bibliography/figure inputs and exact hashes.
- [x] Check the two current figures with their separate pinned renderer; inspect
  them in color and grayscale, including page placement and captions. The final
  label-only redraw reuses a SHA-256-bound verified data manifest; it does not
  reopen raw HTTP receipts or requalify runtime behavior. Rebuild only the
  changed architecture diagram, whose host-staging label is “Bounded payloads.”
  Latest Ghostscript 9.54.0 image review opens color and grayscale standalone
  figures at 180 dpi and final PDF pages 6, 12, 18, 20, 24 and 25 at 120 dpi:
  architecture, Algorithm 3, experiment families, logical-host capacity and both
  new plots/captions. No clipped labels, line/text crossings, table overlap or
  indistinguishable grayscale markers/hatching are found. New plot labels are
  at least eight points; mathematical scripts are not body labels. The earlier
  scoped review of Algorithms 1–2, prior work, abstract and conclusion remains
  historical review evidence, not a fresh every-page manual proof.
- [ ] Complete an independent full manual, every-page submission proof. Scoped
  rendered-page review and automated layout checks do not replace that approval.
- [ ] Complete the final manuscript-wide citation/guarantee and third-party
  distribution/model/toolchain licensing audits. The scoped new SGLang source
  check above is done, not a substitute for full-paper review; positioning does
  not imply a matched benchmark, verified release or tested upstream model path.
- [ ] Obtain author-confirmed declarations, funding, acknowledgments, conflicts
  and submission approval. Do not invent an ORCID, release, DOI or data deposit.

Historical exclusions and supplementary results are indexed in the
[artifact guide](ARTIFACT_GUIDE.md#historical-exclusions-and-provenance), not
reclassified as new evidence. New experiments require fresh plans, destinations
and owner authorization; this checklist is not permission to launch them.
