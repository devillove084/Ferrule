# HTTP cancellation v1

A completed, one-lifecycle, two-request actual35B HTTP cancellation/recovery
experiment. See [PLAN.md](PLAN.md), the immutable machine plan [plan.json](plan.json),
[RESULTS.md](RESULTS.md) and [receipts/result.json](receipts/result.json).

- `client.py`: stdlib SSE/TCP client and matched runtime cancellation evidence.
- `run.py`: private two-request specialization of maintained serial journaling,
  source-role binding, authenticated worker/capture/supervisor, maps and cleanup.
- `test_http_cancel.py`:16 CPU-only tests, including real localhost HTTP closure
  and successful/failed two-slot serial paths. These tests launch no model/CUDA.
- `curate.py`: CPU-only allowlist projection and exact private-raw cross-check.
- `test_receipt.py`: offline publication invariants and credential-leak rejection.
- `STAGE_FILES.txt`: exact source/plan/curated receipt whitelist for the parent.
  Never stage the ignored `private-case-*` directory.

Validation, from this directory:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest -v test_http_cancel test_receipt
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B curate.py check
```

The first command is CPU-only. The second requires preserved private evidence;
it rechecks raw hashes and frozen input roles without executing a native binary,
querying a GPU or issuing HTTP. `run.py` has an exclusive GO/no-replay guard: the
completed case must not be resumed or rerun. There is no new fault-injection flag.

The source inputs were tested and hash-frozen before the single GPU run. Per the
parent's delegation, this agent did not commit, push or touch the Git index; the
parent/archivist owns that serialization. Post-run curation files are separate
from the pre-run frozen implementation and cannot retroactively change its gates.

The public receipt omits raw process/launch authentication and private telemetry.
The machine plan contains local canonical model/artifact paths and the selected
physical GPU UUID, but no credentials. Those identifiers are provenance, not a
claim that its command is portable to another machine. All raw authentication
records are private, retained whole and bound by SHA256.
