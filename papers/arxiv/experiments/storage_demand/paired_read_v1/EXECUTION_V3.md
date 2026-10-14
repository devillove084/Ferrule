# Corrective V3 integration — candidate pending, no execution yet

This append-only integration consumes the existing `plan_v3.py`,
`protocol-v3.json`, `cases-v3.json` and CPU seal. Those eight preregistration
files and all 68 historical files are unchanged. Their declarations describe
what was available at preregistration time; this separate adapter does not
silently change `launch_entry`, bind a future candidate, or reseal that plan.

## Real current build identity, not quality acceptance

The current build owner's ordinary local freeze/artifact receipts identify:

- Directory: `target/validation/fixesmetadataCLIcurrent-quality-20261015-01`
- Full source cut: `86c58a017d7c4b53490f536a3b1a133f2f7848cc`
- Source digest: `10183d246a1278248273e7d3055d1690cb56854cbff30f9f0662f18813dc8188`
- Exact 1484-member archive: `e0d160996147820fcc64cfb29070b265b2297a6d9265a21de541ed77e8f2fa65`
- Staged CLI: `7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01`

`current_binding_v3.py` rejects every other tuple, including the old033519c
candidate and any unknown source or binary. These are actual recorded build
identities, **not** numerical/CLI-serving PASS. At this integration's validation,
`FINAL.json` and `evidence-manifest.json` are absent: all nine completed same-cut
quality gates and their final audit must arrive before binding. No
`actual-build-review-v3.json`, `actual-binding-v3.json` or
`execution-sealed-v3.json` has been manufactured in anticipation.

The reviewer uses an isolated instance of the maintained current-builder review
logic for actual Cargo argv/profile/toolchain/dependency/native/runtime/help
and constructor/full40/fresh/ordinary/cold evidence. Its separate V3 source
review checks the new exact immutable archive, not old-source overlap equality.
Both arms' diagnostic and ordinary reports must match all nine scheduler fields
and unchanged capacity/read budgets. Recorded CLI field-help must include the
explicit cohort flags. First planned actual serving is still pending; neither
separate test binaries nor a build receipt prove HTTP readiness/equivalence.
CPU build custody remains unobserved rather than authenticated by assumption.

## Public versioned CPU entry

From repository root, this works now without GPU, model, HTTP or new binding:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/run_v3.py plan
```

The stdout is one JSON object, includes the explicit versioned execution entry,
and retains the original prospective null candidate and zero actual starts.
`run_paired.py` and `run_actual_v2.py` are unchanged. Execution uses their
maintained finite-arm coordinator and lifecycle worker by a scoped,
exception-safe process-local adapter, not a duplicate engine or process harness.
Worker re-exec points to `run_v3.py --_worker`; candidate/execution checks are
V3-only and the actual maintained server spec consumes `cases-v3.json`.

Every arm issues the entire frozen **12-request** sequence: 4 initial,
1 excluded-first QA, 2 QA warmup, 5 measured QA. Both actual server argvs include
`--decode-cohort-target 1 --decode-cohort-max-deferrals 0`; only read mode differs.
NVTX remains OFF. Seed20261015 order is B/L, B/L, L/B, L/B, B/L. This independent
study retains five blocks, ten lifecycles, 120 requests, 50 measured QA, an
absolute 7200-second study envelope, 650-second lifecycle bound and 100-second
cleanup reserve. There are no replacements or extra warm/probe lifecycles.

## Bind commands — only after the real same-cut nine-gate PASS arrives

These exact commands perform CPU-only evidence review and exclusive-create new
V3 receipts; they do not build, run help, qualify on GPU or start a server:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/current_binding_v3.py review-and-bind --qualification /home/admin/luhuanbing.lhb/Ferrule/target/validation/fixesmetadataCLIcurrent-quality-20261015-01
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/current_binding_v3.py seal-execution
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/run_v3.py check
```

`--qualification` is mandatory for binding. No implicit old/default/future
candidate is selected. Existing receipt paths are exclusive, never overwritten.
A pending, failed, stale, unknown-cut or incomplete candidate is rejected.
Execution seals bind the unchanged V3 plan/cases plus maintained/new adapter
implementation pins, without recomputing old V1 acceptance against new source.

After a real binding, validated execution seal and **separate renewed owner
permission**, the versioned interface is `run_v3.py start|next|finalize
--directory <new paired_read_v1/artifacts-v3-NAME>`. Only `start` additionally
requires `--sole-gpu-owner-granted`. One `next` consumes exactly one planned arm;
failed/incomplete attempts permanently prevent replay. Registration uses its own
V3 schema, boot/deadline, protocol/case/binding hashes and UUID inventory. Old V2
schemas and `artifacts-current-20261015-01` are rejected. Newly produced final
receipts are labelled V3 at exclusive creation; no V2 raw receipt is rewritten.

The maintained lifecycle still owns all-eight locks/idle, baseline/custody,
authenticated subreaper, HTTP actual enqueue-ID joins, source/runtime/NVTX,
positive read-ledger checks, normal cleanup and finite deadlines. Idle/ECHILD is
not a substitute for a missing owner ledger. Missing evidence stays failed or
unknown. The stopped V2 result and its wrong implicit cohort remain unqualified;
no correction or later quality result reclassifies it or repairs its denominators.

## CPU regressions and publication

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/paired_read_v1 -p test_run_v3.py -v
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/paired_read_v1/publication_v1 -p test_publication.py -v
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/publication_v1/check.py
```

The tests use temporary CPU fixtures/mocks, never fake durable candidate PASS.
The safe stage whitelist includes these new adapter source/test/docs, not their
future candidate receipts, executables, maps, stderr or any raw credential-bearing
launch/observer/environment artifact. No Git operations belong to this adapter.


### Handoff blockers observed by CPU inspection

The real build directory is owned by the independent build/GPU owner. Its frozen
source currently has three extra, nonarchived files:

- `frozen-source/scripts/__pycache__/gpu_idle_window_v2.cpython-310.pyc`
- `frozen-source/scripts/__pycache__/run_pinned_jobs.cpython-310.pyc`
- `frozen-source/scripts/__pycache__/test_vllm_qwen35.cpython-310.pyc`

The strict source reviewer rejects this tree, rather than ignoring or silently
adding those files to the source cut. The build owner must remove its generated
bytecode and keep later helper imports bytecode-free before the final exact-tree
review. This adapter has not modified that owner's directory. This does not
change any archived source hash or justify skipping a quality gate.

The broader historical CPU suite currently has four failures (three errors and
one assertion failure) because the live `serve.rs` is now corrected-cut source
rather than the old033519c pinned bytes. In particular, the old V2 execution
validator recomputes its historical V1 source precondition and now fails closed.
That does not mean old receipt bytes or digests were changed, and it must not be
fixed by overwriting the frozen source pins/tests. The new V3 plan/integration,
new exact source identity checks, and offline publication checks do not transfer
that historical live-source admission to the new candidate.
