# Reviewed final-layout adapter for corrective V3

This append-only adapter binds only the immutable `86c58a0` corrected source
archive and CLI SHA `7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01`.
It does not modify the published V2/V3 adapters, old HOLD/failure receipts,
qualification inputs, study plan, production code, manuscript, or history.

`evidence.py` recovers the final-layout readers from the prior non-admitting
`binding-review-v3-20261015-01/review_candidate.py`. Its CPU build checks use the
existing raw command validator; its GPU gates use the existing authenticated
producer and numerical-quality validators. Final per-role native snapshots,
actual build flags, staged-final help, loaded GPU runtime maps, source/archive
hashes, and ordinary/cold same-primary fingerprints are all verified. CPU build
custody remains **unobserved**, not upgraded to authenticated GPU custody.

All 1,484 archived inputs must be unchanged. The only extra frozen-tree files
allowed are the three exact hash-allowlisted, evidence-manifest-pinned Python
bytecodes in `binding.EXTRAS`. These are generated, noncompiled evidence, not
Rust/C build inputs. They and the original manifest are preserved in place;
there is no extension-based exemption or unclassified extra allowance.

`binding.py` separately reviews the real CLI source admission and preserves the
old fixture failures. It binds the real `staged-final/cli` and final runtime pin
manifest. It never claims CLI mathematical equivalence or a clean full CPU
suite. New private bindings are written exclusively under
`bindings-final-v3-20261015-03/`; raw receipts and credentials are not committed.
All five adapter source/test/document files are pinned by review and execution.

Use the repository's CPU-only Python 3.12 with `-S -B`:

```sh
python3.12 -S -B final_adapter_v3/binding.py review-and-bind --qualification /home/admin/luhuanbing.lhb/Ferrule/target/validation/fixesmetadataCLIcurrent-quality-20261015-01
python3.12 -S -B final_adapter_v3/binding.py seal-execution
python3.12 -S -B final_adapter_v3/run.py start --directory /home/admin/luhuanbing.lhb/Ferrule/papers/arxiv/experiments/storage_demand/paired_read_v1/artifacts-v3-final-20261015-01 --sole-gpu-owner-granted
python3.12 -S -B final_adapter_v3/run.py next --directory /home/admin/luhuanbing.lhb/Ferrule/papers/arxiv/experiments/storage_demand/paired_read_v1/artifacts-v3-final-20261015-01
```

The examples run from `paired_read_v1/`; substitute the pinned Python path.
Call `next` at most ten times, stopping permanently on the first failed or
unknown arm. Each call retains the maintained 650-second lifecycle bound and
whole-study 7,200-second deadline with cleanup reserve. The first call is the
first planned arm, not a pilot. UUID locks, birth identity, subreaper ECHILD,
all-eight stable idle, actual HTTP token/enqueue joins, shutdown JSON, and NFS
mountstats retain their existing validators. No retries, replacement pairs,
foreign signals, warmup/probe workloads, cache flushing, or runtime NVTX.
Physical post-drop release remains unobserved, distinct from retained observed
closure and runtime Unknown. NFS counters are timed shared-client coverage,
not backend disk bytes or workload-exclusive attribution.

The earlier `bindings-final-v3-20261015-01/actual-build-review-v3.json` is
preserved as a pre-binding attempt: the historical root-only seal writer
rejected its child-directory destination. No binding or GPU job was issued.
The adapter now uses its own exact-destination exclusive seal writer.

The `-02` binding attempt is also preserved: inherited function default
arguments still selected the old root-level binding at execution sealing.
The new adapter explicitly supplies its own binding/execution paths, with a
regression test. Neither pre-execution attempt created a GPU owner.
