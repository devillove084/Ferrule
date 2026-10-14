# Reviewed immutable CPU CLI build adapter v1

**Build evidence is not quality PASS, candidate binding, GPU permission, or a
fresh freeze.** This additive contract leaves the frozen protocol, case, request
schedule, requirements JSON, historical receipts, and production code unchanged.

## Contract difference

| Contract | Legacy M7 | Explicit immutable CLI adapter |
|---|---|---|
| Selector | No adapter; `ferrule.paper.m7.build.v1` | `build_adapter = ferrule.storage-demand.cuda-cli-build-adapter.v1` |
| Raw build schema | Maintained M7 receipt | Original `ferrule.cuda-only-cli-release-build.v1`, unchanged |
| Validation | Existing `contract.validate_build` unchanged | `runnerhelpers.build_receipts_v1.review_cli_build` |
| Source identity | Existing live-source contract | Full immutable compiled copy and source tar members, relative and original-absolute-key digests |
| Build process evidence | Existing authenticated subreaper required | Bounded CPU timeout, recorded successful Cargo completion; authenticated build custody **unobserved** |
| External inputs | Existing M7 inventories | Full recorded compiler/CUTLASS/previous-dependency/runtime pin groups plus actual dependency-source and native-archive copies |
| Runtime | Run mappings remain required | Recorded CPU `ldd` resolution is checked; model/vendor `dlopen`, actual server maps and GPU context **unobserved** |
| Help | Existing exact-binary field-help receipt | Original GPU-hidden help/serve-help/version argv, stdout/stderr hashes, exact flag tokens |
| Numerical qualification | Separate strict gate | Same-source full40/fresh/cold gate, independent references and authenticated actual producers; not waived |

No receipt schema is renamed, no `status=pass` or missing custody is fabricated,
and no build-time subreaper requirement is invented for a CPU-only builder.
The unconditional `libferrule_cpu_native.a` is not a Cargo CPU-inference feature.
Unknown adapter versions, oracle/CPU-reference features, missing archives/pins,
copy/hash/source-cut drift and false loaded-runtime claims fail closed.
The v1 compiler/SM/toolchain/environment/release policy is deliberately specific;
a different build contract needs another reviewed adapter version.

Post-build checks against previously sealed input pins remain **post-build**
checks. They are not authenticated pre-build capture, a hermetic toolchain image,
bit-identical native code, or producer authenticity beyond the recorded evidence.

## Exact ready API

- `review_cli_build(descriptor, raw_build, case)` returns factual identity/coverage.
- `validate_candidate_build(candidate, raw_build, config, case)` explicitly
  dispatches the adapter or the unchanged legacy validator. Unknown versions fail.
- `protocol.review_build(path, case=None)` accepts only
  `ferrule.storage-demand.cpu-build-review-input.v1`; it returns
  `build_evidence_reviewed_quality_pending_not_authorized`, never a candidate seal.
- `protocol.prepare_candidate(candidate, case=None)` still accepts only the
  existing quality-qualified candidate-binding input. The new adapter is selected
  by its explicit `build_adapter` field; all three current quality gates remain.

A CPU review descriptor uses **absolute local paths** and SHA256 `artifact_pins`
covering every direct evidence file. It needs `schema`, `build_adapter`, `binary`,
`binary_sha256`, `build_receipt`, `source_archive`, `source_archive_manifest`,
`serve_help_receipt`, `build_evidence` and `artifact_pins`. The source manifest
here is the builder's actual `ferrule.cuda-cli-source-freeze.v1` audit, not a
synthetic `sources_before`/`sources_after` map.

For the reviewed builder layout, `build_evidence` has these keys:

| Key | Original evidence file |
|---|---|
| `artifact` | `release-artifact.json` |
| `build_stdout`, `build_stderr` | `newbuild/cli-build.stdout`, `newbuild/cli-build.stderr` |
| `compiler_pins` | `compiler-runtime-cutlass-pin-reverification.json` |
| `dependencies` | `actual-compiled-dependencies.json` |
| `native_archives` | `native-archive-pins.json` |
| `immutable_native_archives` | `immutable-native-archives.json` |
| `runtime_resolution` | `runtime-library-pins.json` |
| `ldd_receipt`, `ldd_stdout`, `ldd_stderr` | `release-ldd.json`, `.stdout`, `.stderr` |
| `help.help` | `{receipt, stdout, stderr}` for `verified-help` |
| `help.serve_help` | `{receipt, stdout, stderr}` for `verified-serve-help` |
| `help.version` | `{receipt, stdout, stderr}` for `verified-version` |

All but the build logs live under `sourcefreezereceipts/`. Paths are supplied,
not guessed by the adapter. The descriptor is owner-authored, not evidence of a
new build. Review reads local receipts/archives/pins; it does not run Cargo,
help, `ldd`, GPU tools, HTTP, `/proc` samplers, or NAS queries. A full review reads
all archived source members and compiled-copy bytes, including preserved fixture
inputs; it is not a lightweight per-request operation.

Repository-root command after the owner supplies a descriptor:

```text
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/newversionedprotocol/protocol.py review-build --build-review ABSOLUTE_OWNER_DESCRIPTOR
```

`ABSOLUTE_OWNER_DESCRIPTOR` denotes an owner-supplied literal path, not a provided
artifact or a command to execute now. `--output` is forbidden for review-build.

## Strict quality/actual-producer relation

The numerical validator uses the real Rust schemas in
`ferrule-composition/tests/unit/engine/model_factory/bf16_full40.rs` and
`bf16_fresh_quality/mod.rs`, and the retained full40/fresh/cold producer checks.
For this fixed StorageMiddle64 configuration the metric/logit denominators are
`804/1,986,560`, `5,427/9,187,840`, and `1,809/3,228,160`. It requires actual
next-token checks, nine-ID common-history/independent-feedback generations,
QA reset/fork evidence, zero failures/nonfinite/quarantine, fixed tolerance,
observer scope, cleanup, budgets, and independent reference manifest hashes.
These are numerical test-executable observations, **not same-CLI HTTP quality**.

The candidate's pinned `quality_source_relation` must retain all existing fields,
plus `authenticated_launches.{full40,fresh,cold}` and
`producer_finals.{full40,fresh,cold}`. Each is the actual producer's
`authenticated-launch.json`: executable argv/SHA, authenticated launch/helper/
worker identity and exec-preserves-PID/birth/validated-lock evidence must join the
successful supervisor receipt's launch/helper/root identity and authenticated
ECHILD. Every test's full compiled-source manifest must equal the CLI cut.
The producer finals are the actual versioned full40 v2, retained-fresh v3 and
cold-owner v4 `FINAL.json` schemas. Each must explicitly accept its wrapper and
join the actual process selector, report hash, executable and source cut. A
successful subreaper or raw numerical PASS cannot override wrapper rejection;
unknown future producer-final versions need review rather than an alias.
A manifest may be the original flat pin map or the actual frozen-source receipt's
`original_source_pins`; live-tree hashes are not substituted for either.
Independent reference manifests, raw report hashes, cold-to-fresh primary relation,
and `supervision=authenticated_drained_all_three` are still mandatory.

**Current `a9fcde90…` remains blocked.** Its 3,533-file cut digest is
`5662b7d2…`. Older full40/fresh results on `fbf0c225…` cannot qualify this CLI;
matching-cut cold results do not convert those reports or failed wrappers to PASS.
The builder's lexical/source-delta relation is not this all-three producer relation.
A read-only review of that build does not bind it or claim it includes `ae65e41`
or later moving model edits. The main owner must freeze the future candidate.

## New-name workflow registration, not automatic resealing

Changed helpers/tests invalidate the original CPU/execution seals. Plain `check`
continues to reject them. `run_pilot.py plan` reports
`cpu_plan_workflow_unsealed_candidate_unbound_not_authorized` and the pin error;
it is not an acceptable seal. `seal-execution` refuses that plan.

The explicit CPU-only `protocol.py seal-workflow --output NEW_NAME` API records a
new `ferrule.storage-demand.cpu-workflow-review.v1` seal **only on owner request**.
It preserves/hashes the original preparation and frozen protocol/case/schedule,
checks every original pin except the two named reviewed implementation replacements
(`protocol.py` and `tests/test_protocol_v2.py`), records old/new hashes for those,
and registers current helper/test/entry/doc pins. No candidate or GPU permission
is recorded. Existing seals are never overwritten and no seal is created by
plan/check/review-build. An unrelated old-pin change still blocks registration.

After review, the owner can supply this new name explicitly using
`protocol.py check --cpu-prepared NEW_NAME`,
`protocol.py bind-candidate --cpu-prepared NEW_NAME ...`, and
`run_pilot.py seal-execution --cpu-prepared NEW_NAME --output OTHER_NEW_NAME`.
Future run/check selects the new execution receipt via `--execution-prepared`;
that receipt contains and revalidates its exact CPU preparation path/hash.
This task does **not** execute either sealing operation against repository files.

Before GPU admission: future immutable candidate freeze, same-cut all-three
quality producers and references, explicit final binding, reviewed new CPU and
execution seals, full Ready/after-request runtime mapping coverage, readable
passive mountstats, renewed sole-owner grant and literal ordered eight UUIDs.
Build-help success supplies none of those missing GPU/run observations.
The five fixed lifecycles, 60 requests, 25 measured QA slots, deadlines, no retries
and no adaptive scheduling are unchanged.
