# Ferrule manuscript workspace

**Ferrule: Bounded Storage-Backed Weight Residency**

English working manuscript, not an accepted submission. Author: Huanbing Lu,
`luhuanbing084@gmail.com`. Unprovided ORCID, funding and author declarations are
not inferred.

## Start here

**[Artifact guide](ARTIFACT_GUIDE.md)** — the single result-to-input crosswalk,
public CPU commands, figure reproduction, source/binary identities, budget and
numerical settings, test selectors, and external-data limitations.

- [Claim–evidence ledger](EVIDENCE_LEDGER.md): what each result supports and what
  remains unproved; intended invariants are separate from observed ledgers.
- [Completion checklist](PAPER_COMPLETION_CHECKLIST.md): scientific, artifact
  and submission gates, without treating documentation work as new experiments.
- [Manuscript PDF](build/main.pdf) and [LaTeX source](main.tex).
- [arXiv preparation CLI](submission/README.md): reviewed source archives and
  isolated compilation checks; browser upload and final author approval remain manual.

The scientific evidence is organized as **quality and controlled budgets →
logical demand and cache pressure → paired scheduling and scoped overlap →
component custody and HTTP cancellation**. Use result labels such as
*paired-study*, *cache-pressure*, *scoped-overlap*, *mixed-fault* and *http-cancel*
in scientific discussion. Engineering paths, hashes and collection histories
belong in the artifact guide rather than the manuscript narrative.

The available results do not establish a physical-memory capacity frontier,
request-exclusive NAS traffic, general latency superiority, complete profiler
capture, or universal fault recovery. In particular, scoped-overlap acceptance
is not full-trace acceptance; the component Unknown test is not a full-model
broker-ledger proof; HTTP cancellation is followed by a fresh request, not
same-request resume. Historical failures remain separate, never pooled.

## Build and verification

From the repository root, the paper build entry point is:

```sh
make -C papers/arxiv
```

For evidence verification, use the [public CPU checks](ARTIFACT_GUIDE.md#public-cpu-checks).
They need no GPU, model weights or network. The five standalone archive suites
pass on Python 3.10.9; the HTTP receipt tests require Python 3.11+ and were tested
on 3.12.4. Do not assume `python` means Python 3.

The [public paired renderer](ARTIFACT_GUIDE.md#paired-public-figures) regenerates
all three paired figures from the verified 623-file archive, without the private
840-input tree. Plotting requires the pinned Python/dependency/font environment;
standard-library receipt verification does not. The
[portable fault archive](ARTIFACT_GUIDE.md#portable-mixed-fault-and-external-build-inputs)
supplies actual component reports, typed evidence and a public-input figure.

The [platform inventory](ARTIFACT_GUIDE.md#post-hoc-platform-inventory) is a later,
same-boot CPU/kernel-visible-memory observation, not a measurement of
experiment-time allocation or installed RAM. Neither that inventory nor the
64 MiB logical staging policy proves physical RAM insufficiency or NFS necessity.

[Native rebuild/GPU re-execution](ARTIFACT_GUIDE.md#immutable-identities-and-data-pins)
requires external executable/source/native/model/oracle inputs and fresh owner
authorization. Current raw serving-source qualification also remains external.
Public hashes and successful receipt/plot checks do not supply those payloads.

Receipt verification does not certify manuscript or submission acceptance.
Final source/PDF consistency, visual review and author approval are separate
[checklist gates](PAPER_COMPLETION_CHECKLIST.md#manuscript-and-submission-acceptance).
Never replay sealed experiment launch commands or publish private raw
environments, authentication or profiler dumps.
