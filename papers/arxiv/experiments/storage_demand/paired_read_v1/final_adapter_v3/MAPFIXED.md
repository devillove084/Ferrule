# Independent map-fixed execution generation

`mapfixed.py` is a destination/seal adapter, not a new lifecycle harness. It
reuses `final_adapter_v3/binding.py`, `run_v3.py`, and the map-evidence worker
introduced in `7d3bee2`. Its only permitted raw study directory is
`artifacts-v3-mapfixed-20261015-01`; private preregistration and freshly reviewed
binding/execution seals live in `plan-mapfixed-20261015-01`. Existing failed
studies and their consumed lifecycles must not be resumed or overwritten.

Before any server start, create the new private plan directory, run the offline
adapter/map tests, commit and push source/tests/docs, then invoke this entry
with `preregister`, `review-and-bind --qualification <immutable directory>`,
`seal-execution`, and `check`. Use the existing `start --directory ...
--sole-gpu-owner-granted`, finite `next --directory ...`, and `finalize
--directory ...` commands through this same entry. Worker re-entry is bound to
this file. No original binding seal is silently reused.

The immutable V3 design, cases, requests, actual-token-ID oracles, seed
20261015, five pairs, ten fresh servers, 120 slots and 50 measured QA slots are
unchanged. Prior V2 and V3 failed denominators remain separate. All admission,
quality, required-library hash/presence/ELF-role, absolute-deadline, all-eight
local-lock/idle, authenticated subreaper, and permanent-STOP gates remain in
force. No extra startup probe, warmup, profiling, retry or replacement is
introduced. The candidate CLI and production source are unchanged.

Raw maps, request/token TRACE, worker metadata and study artifacts are private
and excluded from Git. Mapped required paths and pinned file hashes do not
prove mapped-memory byte equivalence; overlay backing identity can remain
unobserved. Client NFS deltas are not backend NAS traffic. Known owner-local
close/retained residency is not post-drop physical release proof. Estimates
retain the original exploratory n=5 interpretation and failure denominators.
