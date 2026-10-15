# Caption: fault-custody-contract

**Completion evidence governs custody after a real routed-down component fault.**
Two actual Mixed CUDA selectors each run Blocking and Lookahead (four passing
cells) on a deterministic width-16 GatedDeltaNet/GQA fixture, four experts/top-3,
BF16 history/KV and F32 SSM. (a) Tx1 commits frontier 1. Tx2 fails after a real
routed-down submission. With retained completion proof, clean abort returns the
512 B working state and two transient KV slots; committed KV/convolution/SSM
remain bit-exact, and an in-test tx3 succeeds before being aborted. Synthetic
Unknown deliberately discards a successful real consumer proof; two subsequent
independent compute and two upload synchronizations cannot repair owner custody,
and publish/reset/shutdown remain rejected. These fence/API outcomes are
source-and-PASS-backed assertions, not independently serialized error receipts.
(b) Clean returns to 1,024 B default-plus-committed state; Unknown retains 1,536 B
and an active transaction. **Post-Unknown KV/SSM contents are unobserved; only
committed metadata is unchanged.** (c) Every cache bar is free plus charged =
65,536 B. Hatching marks the 6,918 B Unknown **subset** of charged, not an
additional bucket. The separate 318,720 B workspace is not added to these owner
views. Host reads independently close with zero charge; Lookahead wastes one
774 B read per cell. Only clean's later explicit owner drop and allocator trim
observe fixture-requested bytes 0 → 0; neither that observation nor NVML/ECHILD
establishes device-wide release. This is a component contract—not integrated
CLI/broker E5, a 35B model fault guarantee, or physical driver proof loss.

Data and schema:
`papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_fault_v1/`.
Source: `fault-custody-contract.py`; products: `fault-custody-contract.pdf` and
`fault-custody-contract.png`. Blue dotted baseline, teal solid clean and brick
dashed Unknown curves, marker shapes, and hatching also distinguish grayscale
printing. Both schedules' measured state trajectories coincide; shared segments
are deliberately overlaid, not jittered into different values. Logical stage
spacing is not wall-clock timing. The manuscript owner decides placement and
size; no paper source is changed by this deliverable.
