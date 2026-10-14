# CPU closeout — no GPU authorization

- 315 tp2_resident + 42 paper contracts + 24 analysis = **381 passed**, zero skips/failures. Focused suite: 83 passed. Actual argv/cwd/environment/exit/log hashes are in the JSON command receipts; final resident receipt is `full-resident-sealed.json`.
- These are the standard-library paper CPU suites, not Rust/GPU or torch oracle qualification. Existing qualification receipts remain historical evidence, not new GPU passes.
- Exact release: `target/validation/tp2-mixed-allocator-v3-release-20261011-01`; binary `7e1c23a88143cb6455e3c6e10bfe43a567fc97b5fd45745800eae3a1e39a0e73`; handoff `8362e938840955840114c59c8092be19eec0dc3b5c9237f10b39832bdda23b28`. All 96 final builder pins verified. Review manifest registration retained.
- `pre_serving.py` restored byte-for-byte from its authenticated retained snapshot, SHA256 `649f1ca9535ffa721d51f9bc527b08135c348f7a41ce61369c9dba6088599f96`. Its entry version is preserved here. V3 dispatch lives only in `paired.validate_release`, gated by the exact handoff path and hash.
- Updated indexed CPU fixtures retain current-source drift rejection and actual frozen-receipt audits. Their original bytes were reverse-reconstructed from only this session edits, matched against BOTH entry hash and authenticated V3 prepared pin (`70571a7d595ea15459c4c66799334a7c9f990080579cde271bb4ac416e690f27`), then archived. No historical raw data or existing pin documents were rewritten.
- Allocator boundary checks now require both owner ready/shutdown records, 3-GiB arena, no ordinary pressure/allocation failure, no quarantine and zero pending retirement. V3 fields are required only for 7e1c; 9ef is not filled with invented metrics. Pending/Unknown peaks remain null. Summary preserves the assessed boundary report.
- Existing info-only 9ef/7e1c profile policy remains rankperf-off, with actual HTTP clock anchors, event API layer separation and bidirectional PID/correlation/UUID audit. It was not launched; no profiler/tool query was performed.

## Prepared next stage

Actual CPU bootstrap and prepare completed:

- Admission: `papers/arxiv/experiments/tp2_resident/admission/allocator-v3-release-20261011-01/smoke-admission.json`
- Prepared: `papers/arxiv/experiments/tp2_resident/runs/allocator-v3-serving-smoke-20261011-01/prepared.json`
- Slot bindings: the same directory `runs/S1/prepared.json` and `runs/S2/prepared.json`.
- `prepared-verification.json` records real source/admission/binary pins. No launch/execution/smoke receipt exists.

The existing entry CLI below is documented only, NOT executed and NOT authorized:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/tp2_resident/paired.py run --id allocator-v3-serving-smoke-20261011-01 --purpose serving_smoke
```

It intentionally omits grant flags and therefore fails closed. The next agent must first obtain a fresh explicit sole-owner authorization and corresponding owner note; this closeout supplies neither. Revalidate the prepared/source pins before launch. No prepare overwrite/resume. S1/S2 must pass for this exact staged 7e1c before promotion or crossed-cycle/profile admission. Failed smoke is not numerical acceptance and must not be retried under this run ID.

## Evidence

`cpu-closeout.json`, `seal-command.json`, `source-pins-final.json`, `paper-source-changes.json`, `preserved-evidence-verification.json`, `prepared-verification.json`, and `seal-manifest.json` are the final CPU records. Earlier failing logs are retained separately, not relabelled green. `seal.py` records this one-shot verification implementation; it performs no GPU query, owner lock, service or profiler invocation.
