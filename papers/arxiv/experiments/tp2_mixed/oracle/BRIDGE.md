# CPU raw-runtime bridge and lifecycle comparator

## Legacy full40-03 bridge: implemented

`bridge.py` consumes the v2 contract and owner streams described in
`crates/ferrule-composition/tests/tp2_mixed_execution/CANDIDATE_CONTRACT.md`.
Give `--input` the **EVIDENCE root**, not its `candidate/` subdirectory.
Parent `.receipt.json` and `.kv-receipt.json` files are inside `candidate/`;
observed full-vocabulary `.actual.f32le` files are at EVIDENCE root.

1. Wait for the producer to finish/drain. CPU fingerprint the 19 required files:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /opt/conda310/bin/python -B \
 papers/arxiv/experiments/tp2_mixed/oracle/bridge.py --input FRESH_EVIDENCE --inspect
```

2. Pass the returned SHA256 and immutable reference pin. Output must be a new,
separate paper-local directory, not under either input or historical reference:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /opt/conda310/bin/python -B \
 papers/arxiv/experiments/tp2_mixed/oracle/bridge.py \
 --input FRESH_EVIDENCE --input-sha256 RAW_SNAPSHOT_SHA256 \
 --reference papers/arxiv/experiments/tp2_mixed/oracle/runs/full40-20261009-03 \
 --reference-sha256 fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d \
 --output papers/arxiv/experiments/tp2_mixed/oracle/runs/FRESH_CANONICAL
```

3. Run unchanged legacy numerical comparison with the emitted candidate manifest
SHA256 (`FRESH_CANONICAL/manifest.sha256`):

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /opt/conda310/bin/python -B \
 papers/arxiv/experiments/tp2_mixed/oracle/compare.py \
 --reference papers/arxiv/experiments/tp2_mixed/oracle/runs/full40-20261009-03 \
 --reference-sha256 fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d \
 --actual papers/arxiv/experiments/tp2_mixed/oracle/runs/FRESH_CANONICAL \
 --actual-sha256 CANDIDATE_MANIFEST_SHA256 \
 --output papers/arxiv/experiments/tp2_mixed/oracle/runs/FRESH_COMPARISON.json
```

Conversion completion is **not** numerical PASS. A failure after creating output
preserves `manifest.json` with `status=incomplete`; no candidate completion is
published. Missing input/pin errors fail before creating output. Nothing is
written into raw input or reference directories.

### Enforced joins and mapping

- Both `capture-status.json` must prove complete available streams, matching
  byte/event/invocation counts. `required_inventory_complete=false` in raw v2
  does not disqualify conversion: this bridge independently checks completeness.
- Binding first, increasing invocation/order, contiguous payload offsets,
  provisional operator phase, successful `execution_end`, then committed KV
  export. Execution end alone never establishes publication.
- Exact published transaction/session/case/phase joins on parent and KV receipts;
  no arrival-order cross-rank join. Packed positions/query/context must match.
  Owner-local execution generations/state indices are retained, not equated
  across ranks. The replicated page generation and logical KV source must agree.
- KV: matching transaction/source owner/slot/generation/incarnation, native head
  range and local/global layer, plane and page/slot/epoch identity; disjoint
  chunk coverage of **all positions 0..committed_tokens**, including retained
  decode prefix. Chunks can arrive in any position order, but the stream's
  owner-global event order itself must remain valid.
- Geometry: full native N-D shape and contiguous element strides per v2; no
  inference of shape from flat length. F32 state/probabilities preserved. BF16
  host expansions require an exact bitwise F32→BF16→F32 round trip, including
  signed zero; nonfinite observations fail.
- `router` provides raw logits and authoritative integer IDs/u32 F32 weight bits;
  `router.probabilities` provides actual F32 probabilities. No softmax, expert
  selection, weighted output or other model arithmetic is recomputed.
- Raw shared gate comes from `mixed.linear.output` with
  `role=SharedExpertOutputGate`, **not** `shared_gate.output` (already gated).
  `mixed.swiglu.partial` must have `role=SharedExpertDown` (ungated shared output).
- Embedding/final norm/global layer outputs must be observed on both ranks and
  match exactly. Missing replicas are not filled. Local head output is observed
  `mixed.linear.output/OutputHead`, checked against its authoritative row range.
  Actual root full logits are checked against the ordered local gather.
- Global caches are only concatenations/reordering of actual local state:
  GQA/SSM along heads; conv by independent Q/K/V segment ordering. No KV is built
  from newly projected K/V rows. GDN remains a provisional working-state
  observation joined to successful publication, not a claimed separate
  post-publication device read.
- Inputs/positions come from validated packed metadata. The canonical all-ones
  mask is the explicitly supported batch-one/no-padding representation.

Output: old `ferrule.paper.tp2-mixed-candidate.v1`, case safetensors and JSON
reports, exactly **2,178 tensors/case** for full40. The only oracle-only fields
are unstable pre-tie HF `native_expert_ids`. The raw snapshot is hashed again
before completion. Numeric tolerance and canonical selected-ID exactness remain
unchanged (`0.00025 + 0.012*abs(expected)`).

**Identity scope:** candidate checkpoint metadata is bound through runtime's
contract reference hash (the producer validates that checkpoint before CUDA).
The bridge does not reopen large weights or independently prove producer code
execution. Raw producer custody, source/binary pins and scratch admission still
belong to the authenticated runtime receipt; a converted archive alone is not
factory, physical TP2 or performance admission.

## Lifecycle comparator: implemented for defined paper recipes

`lifecycle_compare.py` numerically compares every call for `greedy`, `canary`,
`fork`, `recreate`, `cold-owner`, reusing `expected_tensors`, checked artifacts,
and `tensor_metric`. It is not the former receipt-only draft.

- Reference schema: `ferrule.paper.tp2-mixed-lifecycle-reference.v1`, generated
  by the existing authenticated `launch.py --scenario ...` entry.
- Candidate schema: `ferrule.paper.tp2-mixed-lifecycle-candidate.v1`. Same
  profile/protocol/checkpoint fields as the reference, status complete, `cases`
  matching `suite.cases`, and `suite={plan,events,cases}`. The canonical plan must
  match reference exactly. No new protocol is installed in `protocol.json`.
- One artifact per `call-NN`, stage `call`, full per-call inventory except native
  pre-tie IDs; report `calls` has exactly that one recorded call. Every candidate
  case also has `publication:{file,sha256}` containing actual parent
  `{case:call-NN,phase:prefill|decode,session,transaction,input_token_ids,
  published:true,error:null}`.
- Events follow `multistep.build_plan` exactly. Per-call events have the actual
  `call`, `state_before`, `state_after`, `untouched_sessions`, and epoch. State
  digests must be computed from candidate observations, never copied from the
  reference. `state_after` is checked against captured arrays. Between-call
  digests, fork source/frontier, logical-session identities and recreated epoch
  are checked. Receipt transactions cannot repeat within an epoch.
- Candidate greedy inputs replay **recorded reference** tokens, even when the
  candidate predicts differently; every greedy ID is still compared exactly.

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /opt/conda310/bin/python -B \
 papers/arxiv/experiments/tp2_mixed/oracle/lifecycle_compare.py \
 --reference FRESH_LIFECYCLE_REFERENCE --reference-sha256 REFERENCE_SHA256 \
 --actual FRESH_LIFECYCLE_CANDIDATE --actual-sha256 CANDIDATE_SHA256 \
 --output papers/arxiv/experiments/tp2_mixed/oracle/runs/FRESH_LIFECYCLE_COMPARISON.json
```

## Completed CUDA references and runtime handoff

See `LIFECYCLE_RELEASE.md` and `lifecycle-release.json` (pinned by
`lifecycle-release.sha256`) for six verified CUDA exports / 33 full40 calls,
actual replay inputs and manifest hashes. The frozen selection in
`lifecycle-contract.json` is unchanged. Generation reused the authenticated
paper launcher; these CPU comparators do not launch CUDA.

The runtime contract now documents a separate `ferrule.mixed-lifecycle-raw.v1`
producer and composition-owned `lifecycle_bridge.py` adapter. Use that adapter
for lifecycle raw evidence, then this directory's `lifecycle_compare.py` for
numerical acceptance. The legacy paper `bridge.py` deliberately remains fixed
to the original Hello/Capital two-phase capture.

**Current admission mismatch:** the runtime contract inspected for this release
accepts only the five default `decode_steps=4` plans. The completed paper
selection is greedy4, fork2, canary2, recreate1, cold-owner1 (two processes).
Only greedy4 is currently plan-compatible. Runtime must explicitly admit the
other exact frozen plans and their bounded call/epoch budgets; do not truncate,
relabel, change the reference or loosen the comparator to bypass this mismatch.
No lifecycle production candidate was compared in this reference release.

Fork shared-KV read authority remains runtime-owned. A reference snapshot is not
permission to export an aliased source. Missing/denied snapshots must block,
never be filled from the reference. Provisional GDN is publication-bound;
committed KV requires its actual matching receipt and both owner statuses.
Numerical equality is not physical COW, rollback, quarantine, mid-compute
cancellation or runtime two-process cold-owner proof. The comparator validates
one suite per invocation; distinct runtime cold-owner processes require separate
authenticated runtime launch receipts in addition to two numerical comparisons.
