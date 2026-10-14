# Authorized lifecycle reference selection

Read `lifecycle-contract.json` for machine-readable plans, input actions, call
schemas and plan SHA256s. These are the existing `multistep.build_plan` recipes,
not an alternate executor or altered numerical protocol.

This finite selection has **6 sequential fresh CUDA exports / 33 model calls**:

| Recipe | Decode steps | Exports | Calls/export |
|---|---:|---:|---:|
| greedy | 4 | 1 | 10 (Hello 5, Capital 5) |
| fork | 2 | 1 | 6 |
| canary | 2 | 1 | 5 |
| recreate | 1 | 1 | 4 |
| cold-owner | 1 | 2 separate processes | 4 |

Order prioritizes greedy4 then fork2. Each export is limited to 1700 seconds
inside the existing authenticated custodian, with a 1800000-ms outer tool bound,
256 MiB lazy decoded-weight cache and 512 MiB capture/call (4 GiB cumulative).
Every run uses a fresh directory, shared local owner lock, pre/post idle checks
and proven-drained descendant custody. No historical artifacts are rewritten.

Inputs: Hello `[9419]`; Capital `[760,6511,314,9338,369]`. `kind=greedy` means the
oracle resolves the preceding full-vocabulary argmax and records that exact input.
Fixed recipes:

- Greedy: independent fresh Hello and Capital sessions, each prefill + 4 decodes.
- Fork: Hello prefill, clone left and right at frontier 1; left greedy twice;
  right greedy once then forced `[760]`; drop branches and greedy-decode the
  still-unmodified parent once. HF clones establish independent math, not a
  requirement to abandon runtime COW or retained-source authority restrictions.
- Canary: Hello prefill; interleaved Capital prefill + greedy decode; drop
  canary; resume Hello for two greedy decodes.
- Recreate: Hello prefill + greedy decode; drop state, clear decoded operands,
  construct a new HF model in the same child; repeat with a fresh session.
  This tests numerical recreation, not complete destruction/accounting of all
  process resources and not a cold process.
- Cold-owner: Hello and Capital, each prefill + one greedy decode, in each of
  two distinct authenticated launches. Compare outputs and process identities.

Outputs: 1,169 reference / 1,089 candidate tensors per `call-NN`. The tensor
stage is `call`, and `call-NN.json` records actual inputs, start/end frontier and
next greedy ID. `suite.events` records session, operation and epoch associations.
Runtime must replay those recorded inputs, use actual parent publication and KV
lease/export receipts, and preserve failed/unsupported snapshot operations as
blocked. **Never synthesize a candidate snapshot merely because the oracle has
one.** The comparator does not authorize shared-KV reads or runtime cancellation.

All six exports are now complete and artifact-verified in
`lifecycle-release.json`; see `LIFECYCLE_RELEASE.md` for pins, replay details and
the current runtime decode-count admission mismatch. This is completed CUDA
reference generation, not lifecycle runtime acceptance. All math stays pinned
HF/Torch logical TP2 on one CUDA device; no physical TP2 transport or performance
acceptance claim.
