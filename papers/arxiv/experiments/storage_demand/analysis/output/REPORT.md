# Storage-demand offline receipt analysis

Scope: the fixed `storage-demand-tp1-middle-20261013-01` and `-02` receipts only. No GPU queries, launches, production imports, or raw-receipt edits.

## Result

- -01: planned 4, dispatched 0, successful 0, not run 4, usage output tokens 0, invocations 0; failed before server launch: `AttributeError: 'types.SimpleNamespace' object has no attribute 'DEVNULL'`.
- -02: planned 4, dispatched 4, successful 4, not run 0, usage output tokens 26, invocations 30.
- Source consistency: **PASS**; 41 bound input hashes and 39 consumed receipt pins checked; 0 divergences.
- Raw-log invocation deltas match per-request and whole-run receipts when consistency is PASS. Gauge checks use last values, never sums. Sampled memory and phase maxima are recomputed from raw samples.
- -02 totals: checkpoint_read_calls=69114, checkpoint_read_bytes=36240064128, h2d_copies=88763, h2d_bytes=36240248688, expert_uploads=11519, expert_evictions=11455, artifact_uploads=34557, expert_hits=0.

## Quality scope correction

- Hello2 derived quality_scope: **HTTP completion and usage checks only; no reference comparison**. Its reference is null and reference_quality_passed is null (not a frozen-reference PASS).
- The frozen Hello2 receipt inherited a shared quality_scope label mentioning frozen HF generation8. The generator normalizes derived metadata from reference presence, not that label; all frozen receipts remain unchanged.
- Only 3 canaries (canary-qa, canary-arithmetic, canary-chinese) have visible-text/eight-output-usage reference checks; 3 pass those limited checks. Hello2 accounts for 1 HTTP-only successful request. No exact generated IDs were observed, and none of these checks establishes numeric acceptance.

## Caveats

- One successful instrumented run (n=1), not independent repetitions; no statistics or confidence intervals.
- ferrule_performance DEBUG overhead is unmeasured; these latencies are not a fast baseline or comparable vLLM throughput.
- Read calls/bytes are logical checkpoint-reader/syscall counters; physical NAS/NFS IO and OS page-cache state are unknown.
- H2D/artifact counters are facade activity, not physical bus traffic or device-kernel timing. Startup IO is not covered by invocation sums.
- Expert uploads are not cache misses. Misses and IO credits are unobserved, not inferred from hits/uploads.
- SSE text/usage matches are not generated-token-ID or logits evidence; frozen reference IDs are never observed generated IDs.
- TTFT is send-to-first-nonempty-SSE content; E2E is send-to-EOF; TPOT is the recorded content-event/usage proxy, not exact token time.
- Cache occupancy/scratch/owner-peak fields are gauges: last invocation and observed maxima, never additive deltas or a final ledger.
- Memory values are independent sampled component maxima, not an additive decomposition, exact peak, hard cap, or continuous coverage.
- Anonymous is resident anonymous memory, not PssAnon. PssAnon/PssFile and physical host cap are unobserved.
- GPU memory is selected-device aggregate telemetry; owned-process attribution and absence of foreign interference remain unproven.
- After-first-request samples include later canaries, not steady state. Shutdown resource_contract_complete is unknown despite exit/custody/idle evidence.

## Reproducibility

`TABLES.md` provides descriptive manuscript-ready tables; exact values are in `summary.json`.
The JSON binds relative input paths to SHA-256 (actual and sealed values). `../inputs.json` seals all consumed sources, including report receipts that cannot self-pin. Existing report artifact pins are independently checked for consumed leaves. Absolute receipt pin names are normalized only at the fixed storage-demand run path; no binary, checkpoint payload, or production-source rehash is implied.
The generator and input-manifest hashes are also recorded. No new timestamp, mtime, host metadata, or hardware observation enters the output. `--check` recomputes all artifacts and rejects byte-level summary/table/report divergence without overwriting them.

From the repository root:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/analysis/tests -v
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/analysis/summarize.py
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/analysis/summarize.py --check
```
