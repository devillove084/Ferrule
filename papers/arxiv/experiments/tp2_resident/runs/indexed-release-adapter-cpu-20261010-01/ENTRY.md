# Indexed qualification review adapter — CPU-only handoff

Status: actual indexed qualification audit PASS; complete current paper suite 277/277 PASS. No GPU query, build, smoke, promote or A/B execution performed.

## New actual CPU entry

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none PYTHONDONTWRITEBYTECODE=1 /home/admin/wxw403883/anaconda3/bin/python3.12 -S papers/arxiv/experiments/tp2_resident/optimized_release.py audit-qualification --handoff papers/arxiv/experiments/tp2_resident/runs/decoded-arena-indexed-lru-20261010-01/handoff.json --handoff-sha256 514b0d8a7f5e05abf958c5e278d13f026d5bda9c4f4191e51e3e9a7e2fc56cf3
```

API: `decoded_cache_release.audit_qualification(link)`. The indexed schema is `ferrule.paper.indexed-lru-handoff.v1`. This returns authenticated test qualification only, explicitly NOT a serving admission. Feeding this handoff to the serving audit/bootstrap remains rejected.

The audit verifies all 23 named gates, actual receipts/comparators, 37 calls/40293 exact, 38 pinned CPU selectors, strict CUDA Clippy/check, current indexed sources, tools/libraries, final historical idle/release, and CPU benchmark provenance/scope. Benchmark timings are not E2E acceptance.

## Historical binding fix

Unindexed and indexed qualification plans are loaded from their own real `plan.json` files and checked against their exact prepared.json pins. CPU selector/argv/receipt inventories are checked against those plans (33 old, 38 indexed), not the mutable current runner manifest. GPU gate names/selectors remain strict.

The old ae3e release test now exercises the actual frozen baseline audit: authenticated historical build-source map files and serving snapshots/receipts, preserving historical source identities instead of requiring today's indexed tree to equal ae3e. Current-tree release admission remains a separate strict audit; no old GPU pass is inherited by AFTER.

Before adapter edits, the three indexed-qualification-pinned adapter/test files were copied to `history/pre-indexed-release-adapter-20261010/` and their exact original hashes are checked. No old qualification, release, test-plan, source-pin map, or historical failure receipt was rewritten. The original qualification full-suite-failure statement remains historical; the new green CPU suite is recorded separately here.

## CPU validation

`tests-before.log`: reproduced original 271 tests / 2 failures + 1 error.
`tests-after-01.log`: original 271 tests all PASS.
`indexed-tests.log`: 6 new schema/hash/plan/source-drift/benchmark-scope tests PASS.
`tests-after-final.log`: full 277 tests PASS, no skipped failures. Mocked lifecycle tests may print intentional fixture failure/launch messages; no real GPU or HTTP server was started.

## Serving / prepare blockers

At this handoff the builder directory `target/validation/tp2-mixed-arena-indexed-lru-release-20261010-01` has preparation/preflight artifacts, but final serving handoff/hash, release receipt and handoff-file-pins have not been published. Final binary SHA and canonical builder production-delta binding are therefore pending. No serving-schema layout has been guessed or admitted.

After final owner-reviewed builder handoff arrives: review its actual serving schema and receipt fields, connect it to this indexed qualification audit, verify the exact two-file production delta and build/runtime pins, then use existing optimized_release bootstrap and paired serving_smoke prepare. Actual SAME-binary S1/S2 must PASS before promote and arena_index_ab admission.

Then the existing arena_index_ab CLI/API prepares separate cycle 1 BEFORE→AFTER and cycle 2 AFTER→BEFORE. Same 3-GiB arena/workspace4096/device22528 and all prior identity/budget/oracle/custody requirements remain unchanged. No executable AFTER prepared receipt was fabricated. Await a subsequent explicit GPU-owner authorization.
