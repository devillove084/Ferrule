# HTTP admission: approved identity issuance and terminal refusals

Review-ready orchestration only. No real identity anchor, bridge approval or
MAIN-GO is authored by this change. No GPU experiment, build, archive operation,
historical recovery or sealed-file rewrite is performed. The independent study
deadline and **180s static / 650s active / 610s reserve** remain unchanged.
Request work/order, phase-W denominators, target/supervisor/cleanup quotas and
strict authenticated all-eight idle remain unchanged.

## Source trace and lifecycle

Originally the 650-second clock began before UUID locks, pre-idle and repeated
layout/candidate/build/source/dependency/four-cell acceptance validation. The
unchanged reserve `530 + 28 + 20 + 32 = 610` left 40 seconds for active setup.
This identifies the charged call path, not attribution of the failed run's
reported 63.480 seconds; production preparation I/O attribution is unproven.

Finite static preparation now precedes the active clock and UUID locks. Final
admission setup is under that same static deadline. Attempt admission records
the actual active start, unchanged absolute outer deadline, and total preparation
time separately from request throughput. Worker/capture retain the original
worker-computed target cap, even if content verification consumes its reserve.

## P1: complete coverage is necessary, not proof of verification

The earlier complete-coverage correction still trusted self-attested current
stat identities. A newly fabricated receipt could copy approved hashes and the
current identities of already-mutated readonly files. Matching such a receipt
was not evidence that those bytes had ever been verified.

The fix adds one prospective candidate field and one explicit MAIN-GO subject
field. Historical candidate v2 subjects are unchanged. Illustrative fields only:

```json
{
  "candidate": {
    "schema": "ferrule.storage-demand.http-transfer-candidate-binding.v3",
    "verified_artifact_identity": {
      "path": "/absolute/review/verified-identities.json",
      "sha256": "SHA256_OF_THE_REVIEWED_IDENTITY_MANIFEST"
    }
  }
}
```

The eventual reviewer-issued MAIN-GO subject additionally contains:

```json
{
  "verified_artifact_identity_sha256": "THE_SAME_EXACT_MANIFEST_SHA256"
}
```

The actual candidate reference and actual purpose/layout/workflow/evidence-bound
MAIN-GO are independently reconstructed and checked. An anchor reference or
identity map placed only in a preparation receipt is ignored as authority.
The anchor loader also checks the actual GO schema, decision, full subject and
explicit anchor hash before returning issued identities.

### Issuance, before MAIN-GO

The existing CPU `review --layout ...` entry has an explicit
`--verified-artifact-identity-output` option. It calls full maintained candidate,
build/source/dependency/runtime and historical acceptance content validation,
then independently derives the complete required file/hash and inventory graph.
Every issued row is captured around a successful actual SHA-256 check:

1. Capture canonical target/device/inode/mode/size/mtime/ctime/link identity.
2. Hash the actual file and compare with the independently required hash.
3. Require the post-hash identity to equal the pre-hash identity.
4. Recheck all file identities and required inventories before exclusive output.

The immutable output schema is
`ferrule.storage-demand.verified-artifact-identity.v1`, containing the exact
scope subject, `verified_inputs` path -> hash/identity rows and readonly content
`verified_directories`. Its status is pending MAIN-GO, never approval. The issuer
requires an unanchored prospective candidate with launch_review null and never
refreshes an already approved anchor. It uses the existing 180-second finite
review cap; no signer, registry, thread, cache service or new runner is added.

The reviewer must independently inspect the actual issuance/content/universe
once before approving the next GO. No real issuance operation was run in this
coding task; issuance tests use tiny temporary component files.

### Cached validation

- Candidate/CLI identity, exact required coverage and MAIN-GO remain independently
  derived from the real layout and hash-bound manifests, not cached PASS flags.
- A readonly content hash may be reused only if its approved anchor row has the
  exact required hash **and** its issued identity equals the current identity.
- Missing anchor, absent issued row, changed identity, writable files and live
  input/workflow/control files require actual byte hashing against the approved
  hash. Wrong content rejects even if the fresh receipt copies its current stat.
- A changed identity with still-correct bytes may pass only after a bounded
  successful local rehash. The anchor is never rewritten or refreshed; subsequent
  validations must rehash again. No process-global memoization is introduced.
- Receipt file/hash sets, tree/native-archive sets and directory coverage still
  require exact equality with the independently derived complete universe.
- Existing stricter phase timers are reused, never refreshed. Capture passes its
  original target deadline through adapter and freshness checks. Missing-anchor
  fallback may exhaust the static/setup reserve and refuse, never weaken checks.

Only immutable content-tree directories are anchored. Current layout,
registration, output anchor and upcoming MAIN-GO are not identity scopes. Anchor
and GO outputs are prohibited inside content-tree/native-archive scopes, so
creating the expected review/GO files does not invalidate an anchored parent
directory. Mutable directories still require complete inventory checks.
Checkpoint shard payloads remain excluded: their existing stat-identity contract
is retained, and checkpoint_content_sha256 stays null.

## Prospective workflow bridge

The default acceptance gate remains exact-workflow strict. Explicit
`candidate.acceptance.workflow_compatibility` permits only the versioned,
separately human-reviewed prospective bridge. It binds the exact original FINAL
`6f544f9d25bc64b6c37ce0f0ffbe2f8e48ad332d80b38ea5b3221905dd530356`,
original layout/workflow/MAIN-GO, new workflow/test pins and unchanged
CLI/runtime/full compiled source/profile/config/quota/input/ref-ID/EOS/producer
semantics. Original bytes are checked against original Git identities, never
relabeled frozen controls.

Only named orchestration functions in the three named orchestration files may
change, including the now-terminal next_trial admission check. Protected original
quality/auth/target-deadline functions and configuration globals remain guarded.
Only the two named CPU regression files may be added. Human review still owns
the changed orchestration boundaries; these guards do not issue approval.

The bridge precedes identity issuance: its prospective subject normalizes away
the future anchor reference and candidate schema addition, just as it excludes
future MAIN-GO fields. The actual new MAIN-GO then binds both exact bridge and
identity-anchor hashes, avoiding a hash cycle without relaxing historical scope.

The CPU `plan --acceptance-final ...` entry reuses the existing c3db CLI and
reviewed EOS references, with eight starts/96 requests and launch_review null.
`compatibility-plan` emits review null. No four-cell rerun or rebuild is needed.
A CPU regression revalidates the actual four historical cells under original
identities with archives/GPU/worker/server execution blocked; only the already
reviewed compiled-build semantic result is a component substitute.

## P2: refusal itself is a permanent terminal barrier

Before direct preparation, public next admission and next_trial discovery, any
root `static-refusal.json` or `static-refusal-*.json` entry blocks permanently.
The barrier does not parse/trust the journal, so unknown-trial, partial and
malformed refusal evidence cannot become a retry. It does not depend on STOP
writing having succeeded. Existing refusal bytes and their original cause are
never overwritten on a later public next attempt.

Earliest registration/boot/trial/alarm/exclusive-path/context-entry errors still
attempt independent STOP/refusal journals and preserve the identical primary
exception if a write fails. Unknown trial/deadline fields stay null. If both
writes are impossible, durable evidence is not claimed. No error is healed into
qualification or inferred ECHILD/physical release.

Active failure cleanup remains bounded: post-idle while authenticated locks are
held, strict idle for positively proven never_spawned, release/reprobe in finally,
and append-only coordinator failures preserving the original assessment.

## CPU regressions and limitations

Tiny synthetic anchor fixtures exercise real issuance/hash/GO-subject/freshness
code; build semantics and candidate graph are explicit component substitutes.
Their temporary GO/anchor fixtures are not approvals of the real CLI. Tests cover
correct candidate/GO/complete fresh receipt after readonly architecture_target.rs
mutation, missing anchors, unchanged anchored reuse, forced live-control hashing,
unchanged-byte fallback without identity refresh, unapproved anchor substitution,
pre/post-hash mutation, immutable-directory output refusal and real timeout.

STOP-only write failure followed by a second public next must reject before
registration reads, UUID locks or supervisor entry while preserving old refusal
bytes. Partial and known-trial journals also block direct preparation/discovery.
Existing clock, target-cap, mutation, custody and sticky-cleanup regressions remain.

From `papers/arxiv/experiments/storage_demand/paired_read_v1`:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest \
  test_preparation_authority test_deadline_phase \
  test_http_transfer test_http_transfer_metrics \
  test_http_transfer_controls.InitialEndpointTests \
  test_http_transfer_controls.CaptureEntryTests \
  test_http_transfer_controls.ProspectivePlanTests \
  test_http_transfer_controls.HTTPReferenceControlTests \
  test_http_transfer_controls.ObserverAcknowledgementTests \
  test_http_transfer_controls.CoordinatorFailureTests \
  test_http_transfer_controls.OwnedCPUStopTests
```

From repository root:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover \
  -s scripts/tests -p test_gpu_idle_window_v2.py
```

Real large build/archive/anchor issuance and checkpoint payload hashing are not
run. Old acceptance, stopped pilot/all 96 unissued slots and separate immutable
historical recovery are untouched. The existing `/papers/` rule ignores both
regression files and this note; they remain unstaged and require explicit
inclusion by commit tooling.
