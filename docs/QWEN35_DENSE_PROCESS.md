# Dense Qwen3.5 process pipeline

The production serving factory implements admission and construction for the strict **Qwen3.5-0.8B dense,
full-depth text-only** profile with **PP2, TP1, EP1, DP1, SP1, CP1**. Select
processes using the existing `--rank-backend process`; no separate model runner
or process-specific forward path is used. The real 0.8B checkpoint now has a
**process-PP2 GPU smoke pass**: readiness, one completion, and normal shutdown.
This is not a full-logit quality oracle or a performance-matrix acceptance.

## CLI

For a CUDA-enabled binary and an existing strict HF checkpoint:

```sh
ferrule serve /path/to/Qwen3.5-0.8B \
  --engine pipeline --backend cuda \
  --pipeline-parallel 2 --rank-backend process --devices 0,1 \
  --rank-timeout-ms 120000 \
  --ctx-size 128 --max-active-sequences 1 \
  --max-batch-tokens 8 --prefill-chunk-size 8 --kv-cache-mb 64
```

`--engine auto` also resolves this topology to the hybrid pipeline. CPU-reference
builds with the opt-in `cpu-reference` feature can use `--backend cpu` without
`--devices` (the current CLI default/auto selects CUDA). These commands remain subject
to strict checkpoint validation and live aggregate host/device admission; a
successful metadata plan is not a readiness receipt.

Each child holds its segment, compact GQA pool and GDN working-state journal.
The parent owns the only logical page manager and transaction/publication
coordinator. Fork/COW, rollback, retirement and release use the existing pipeline
protocol. Absolute deadlines do not slide on progress; unknown completion is
sticky, and neither failed work nor cleanup is replayed on a replacement process.
Wire v4 requires matching parent/child builds.

## Frame and memory admission

Before any child starts, the factory bounds full-vocabulary output for **every
admitted row**, rather than assuming last-token logits. Negotiated parent frames
grow only as required; the CLI child accepts the existing **128 MiB hard ceiling**.
Ordinary standard/numeric parent limits are unchanged. Custom child executables
must accept the negotiated limits too.

Sizing reserves 32 JSON bytes per float plus metadata and envelope overhead.
For vocabulary 248,320, eight output rows require a conservative payload bound of
63,574,016 bytes; 16 rows fit the frame ceiling, while 32 rows fail preflight.
Use explicit batch/chunk limits as above. The factory does not silently clamp a
batch to make IPC fit. Logical page metadata also counts against the frame bound.

Host admission includes the existing model/working-state/KV ledger plus all
simultaneously retained child catalogs, process overhead, JSON DOM/frame copies
and parent staging. These conservative charges can be substantial; a frame
ceiling is not an instruction to preallocate that much memory, but a smaller
currently observed frame does not waive admission. CUDA stages additionally
recheck their rebound segment budgets against live device capacity.

## Explicit exclusions and validation

No production GPU MoE hybrid PPxEP, TPxEP, process TP, restart/replay, vision/MTP execution,
partial-depth execution or partial recurrent-state retention is enabled by this
path. The separate [CPU-reference parallel composition](PARALLEL_COMPOSITION.md)
now includes the actual PP2×TP2×EP2 runtime factory, using the existing hybrid loop,
single parent coordinator and original owner journals. Its local-math/global-protocol
test split and retry/Unknown contracts do not enable production GPU composition or
process TP, and are not a claim that the final tests have passed. The resident-root
numeric EP path is separate.

Initial implementation validation used real CPU-reference child processes with a
tiny independently oracled hybrid fixture, strict-profile metadata/CLI rejection
tests, and compile-only CUDA checks. The later current-owner run adds real
checkpoint GPU smoke evidence below; it does not replace the tiny fixture oracle
with a full-size numerical or performance acceptance.

### Evidence checkpoint: 2026-10-02

The [CPU reviewer report](../target/validation/recovered-integration-cpu-review-20261001/REPORT.md)
and its `test-reference.log` record passing real-child tests:

- `hybrid_production_factory_executes_real_children_and_session_lifecycle`;
- `process_hybrid_pp2_prefill_decode_fork_cow_and_release_match_oracle`;
- `process_hybrid_executed_working_state_rollback_keeps_source_pristine`;
- `lost_hybrid_child_retains_unknown_and_never_replays_rollback`.

These tests use the tiny CPU-reference fixture. Separately, the
[current-owner report](../target/validation/qwen35-35b-current/current-owner/REPORT.md)
and [dense smoke result](../target/validation/qwen35-35b-current/current-owner/dense-pp2/result.json)
record a real `/mnt/nas1/hf/Qwen3.5-0.8B` CUDA process-PP2 run on devices 0/1
with the CLI limits shown above:

- Startup/readiness **10.2749 s**; one non-streaming `Hello` completion.
- Actual input **1** / output **8** tokens; request **1.1734 s**, finish `length`.
- F32, full24, PP2/TP1/EP1; normal exit0, port18356 released, devices0/1 idle.

This is a bounded end-to-end smoke, not a throughput benchmark, complete lifecycle
fault matrix, or full-logit quality oracle. It is not the separate 35B process-EP8
run. Root/workspace **19/19** and current HostKV live **2/2** also passed in that
owner's report; the latter does not enable transaction-append DMA dispatch.

Full-logit numerical acceptance and a broader lifecycle/performance matrix
remain separate gates; later host-only lint checks do not revalidate the smoke
on a rebuilt binary. Current CUDA thread MoE compositions and W8/Mixed quality
are separate profiles, documented in [Qwen3.5](QWEN35.md) and
[current acceptance](CURRENT_ACCEPTANCE.md). This dense F32 smoke cannot close
W8A16 strict quality failure or authorize process TP, numeric process PP×EP
or append DMA.

Relevant implementation:

- `crates/ferrule-composition/src/model_factory/hybrid_pipeline.rs`
- `crates/ferrule-composition/src/model_factory/hybrid_process.rs`
- `crates/ferrule-runtime/src/parallel/process/decoder_hybrid.rs`
- `crates/ferrule-cli/src/commands/serve.rs`
