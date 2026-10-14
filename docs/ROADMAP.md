# Ferrule roadmap

This document contains unfinished work only. Each item is intentionally phrased
so it can become a GitHub issue or project card without carrying implementation
history. Implemented support and pinned runs are indexed in
[CURRENT_ACCEPTANCE.md](CURRENT_ACCEPTANCE.md); an untested profile must not
inherit a neighboring profile's acceptance.

## B300 correctness

- Prove full-model, end-to-end DeepSeek-V4 Flash token parity on B300.
- Complete B300 kernel, graph, stream, lifecycle, and failure-path validation.
- Complete CPU reference bit parity for every retained DeepSeek-V4 attention
  boundary.
- Validate long-context compressed selection beyond current release limits or
  define a checkpoint-faithful bounded alternative.
- Add official nonzero-temperature Gumbel-max sampling if sampling parity is
  required.

## Performance and concurrency

- Complete a source-pinned, same-resource Ferrule/vLLM serving comparison with
  explicit arithmetic/topology compatibility, independent paired process runs,
  request/failure records and confidence intervals. Historical mismatched rows
  and operator timings do not complete this gate.
- Rebuild and validate public Mixed serving on sources matching the latest
  attention quality result; the earlier Hello lifecycle used a different binary.
- Extend Mixed quality beyond the bounded three-prompt/token-budget gate without
  changing references or silently widening tolerances; retain W8A16's separate
  failed full40/task-quality gates.

- Pass reproducible B300 multi-request serving acceptance.
- Meet the documented throughput and latency gates with confidence intervals.
- Prove useful compute, I/O, and upload overlap from causal timelines.
- Tune proposal width and expert residency without weakening exactness or hard
  resource admission.

## Out-of-core resources

- Complete dense and static parameter materialization through the unified
  resource path.
- Complete mutable KV, activation, gradient, and optimizer-state lifecycles.
- Enforce transient upload headroom and global resident-expert byte admission.
- Validate working sets larger than device memory under sustained load.

## Distributed execution

- Extend beyond the existing local standard and strict Qwen3.5 thread/process
  profiles: multi-node transport and production DP/SP/CP need separate support,
  placement, admission, communication and failure-custody validation.
- Implement and validate distributed BF16/Mixed arithmetic and wire/state
  contracts before admitting those profiles; single-owner Mixed PASS is not a
  distributed precision permit.
- Add CUDA process TP or numeric hybrid process PP×EP only with their own
  bounded transport, publication and failure-cleanup proof. Existing resident
  process EP and dense 0.8B process PP2 do not provide that proof.
- Extend strict full-model acceptance to currently construction-only degrees
  such as Qwen3.5 TP4; the four accepted F32 topologies remain separately pinned.

## Training and RL

- Extend resolved stages and terminalization to forward, backward, and optimizer
  execution.
- Implement checkpointed activation, gradient, and optimizer materialization.
- Add post-training and rollout-learner transaction protocols.

## Portability

- Extend Qwen/`sm_86` validation beyond the pinned accepted profiles, workloads
  and binaries; complete the remaining portability and failure-path coverage
  without relabeling existing bounded end-to-end passes as unimplemented.
- Validate `sm_121a` independently from other CUDA profiles.
- Add Metal, ROCm, and Ascend providers behind the same semantic plans.
