# Ferrule roadmap

This document contains unfinished work only. Each item is intentionally phrased
so it can become a GitHub issue or project card without carrying implementation
history.

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

- Define topology and placement for data, tensor, pipeline, expert, sequence,
  and context parallelism.
- Integrate communication stages, buffers, fences, credits, and cancellation
  into execution transactions.
- Validate deterministic distributed publication and failure cleanup.

## Training and RL

- Extend resolved stages and terminalization to forward, backward, and optimizer
  execution.
- Implement checkpointed activation, gradient, and optimizer materialization.
- Add post-training and rollout-learner transaction protocols.

## Portability

- Restore Qwen end-to-end validation and complete the `sm_86` portability gate.
- Validate `sm_121a` independently from other CUDA profiles.
- Add Metal, ROCm, and Ascend providers behind the same semantic plans.
