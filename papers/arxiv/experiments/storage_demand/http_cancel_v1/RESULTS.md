# One actual HTTP cancellation / recovery lifecycle

**Completed; no retry, smoke, profiler or extra generation.** The preserved CLI86
ran one full40 35B Mixed Lookahead lifecycle, with the exact middle64/320MiB case,
NVTX off and unchanged production/source/model inputs. Total coordinator envelope:
**64.6082s**, within the frozen270s cap. CPU preparation initially caught a Python
environment-key merge error before any GPU query/launch; it was fixed and covered
by a regression test before sealing the protocol. There was exactly one GPU run.

## Actual HTTP and runtime outcomes

| Observation | Cancel QA8 | Fresh recovery QA8 |
|---|---|---|
| Actual HTTP completion ID | `cmpl-ferrule-1` | `cmpl-ferrule-2` |
| Worker request/session | `1 / 1` | `2 / 2` |
| Client outcome | Controlled censor after first nonempty SSE content | Complete through usage, length and DONE |
| Client content | ` Tokyo` | ` Tokyo.\nThe capital of Japan is` |
| Actual worker enqueued IDs | `[25358]` | `[25358,13,198,760,6511,314,6124,369]` |
| Client nonempty content events | 1 | 8 |
| Runtime terminal | `cancelled`, `cancellation_submitted=true` | `finished`, `max_tokens`, no cancellation requested |
| Client usage | **null, expected** | prompt5 / completion8 / total13 |
| Worker terminal usage | prompt5 / completion1 / total6 | prompt5 / completion8 / total13 |
| Terminal enqueue | `cancelled`; closed channel, enqueue=false, usage unavailable | `finished`; enqueue=true, ordinal8 |
| Quality result | **Not applicable; not a QA success** | Exact frozen eight-ID/text/usage pass |

The client shut down TCP approximately **63.4µs after observing the first nonempty
content event**, before logging/sampling/validation. The matching runtime terminal
was passively observed after **0.6680s**, well under45s. The HTTP terminal was
`cancelled/client_disconnect`; the worker also recorded cancellation requested
and submitted, with neither cancellation failure nor restore/shutdown required.
This was not a natural max_tokens completion race.

Whole-log coverage: two admissions, two HTTP admissions, two worker terminals,
two terminal enqueue attempts, two HTTP terminals, nine successfully enqueued
token IDs, eleven decoder invocation summaries and one owner shutdown summary.
Worker IDs are **successful enqueue IDs**, not retokenized text or consumer token
IDs. The one observed client content event does not itself prove a token boundary.
Recovery was issued only after cancellation acceptance and was a new request and
session in the same resident process, not a resumed request.

## Logical resources and limits of observation

The cancelled request's serial log window contains two completed invocations
(the callback/worker can finish a call while the disconnected request is being
cancelled); recovery contains nine. Their recorded logical reader totals are:

| Serial window | Reader calls | Reader bytes | Expert uploads | Evictions |
|---|---:|---:|---:|---:|
| Cancel QA8 | 8,628 | 4,524,109,056 | 1,438 | 1,374 |
| Recovery QA8 | 22,068 | 11,571,399,936 | 3,678 | 3,678 |

These are logical checkpoint-reader and serialized log-window observations, not
NAS backend bytes, aborted physical reads, or a transaction-to-request ledger.
Owner-lifetime reader statistics report5,116 verified/completed reads and zero
cancelled/failed/unknown reads: **request cancellation is not evidence that an
individual synchronous checkpoint read was interrupted**.

Final actual owner evidence:

- Closure: `known_closed_owner_local`; runtime/physical close observed `closed`,
  physical quiescence verified, residency observed **retained**.
- Expert cache:64 entries,201,351,168 resident bytes,8,388,608 scratch bytes,
 209,739,776 charged bytes; peak219,754,496 bytes (under335,544,320).
  Pending upload and unknown quarantine bytes observed0; quarantined=false.
- Reader:closed;440 sequences started/finished; final logical host charges0,
  no pending/unknown read slots. Cumulative/peak values are not summed over windows.
- Shutdown runtime pending operations observed0. This is **not a before/after
  per-request pending ledger**, which remains unobserved/null.
- Shutdown KV broker domain:capacity32 slots, in_use0, high_water1. Actual
  per-request KV/frontier/epoch snapshots remain **null**. Final default model
  state reports64,880,640 resident bytes and unpoisoned generation0/position0;
  that is not a per-request KV release or same-request continuation proof.
- Final physical release, allocator detail and runtime fence remain unobserved.

The receipt includes the first/last invocation gauges for each serial window and
the preceding completed observation where available. No initial unobserved gauge
is filled with a fabricated zero. Invocation transaction numbers are retained as
observations, explicitly **not** joined to a request identity.

## Identity and cleanup

All12 required pinned CLI libraries were observed at Ready and after requests,
with matching pathname hashes and maintained inode/role checks. Raw maps were
saved before validation. Each snapshot retained one permitted deleted
nonlibrary mapping and seven explicit overlay-device aliases; live mapped-memory
hash and overlay backing identity remain unobserved.

The maintained direct-child shutdown/capture returned0. Authenticated supervisor
ECHILD, absence of recorded owners, V2 three-consecutive all-eight idle samples,
and all-eight lock release/nonblocking reacquisition **all passed**. The resource
observer reported `observed_within_targets`. No foreign PID signal was issued.
Known closed retained resources and idle do not prove universal physical release.

## Scope and provenance

This is **normal production HTTP cancellation followed by fresh-request recovery**.
It supports no stale continuation observed in the frozen recovery eight tokens.
It does not prove same-request resume, arbitrary KV equivalence, Unknown recovery,
a complete integrated transaction/fault ledger, or general fault tolerance.
No production, native, model, server, runtime, manuscript or other protocol changed.

- Frozen plan SHA256:
  `95b50c2c7adf360a8ef22e720e9d0c5e3d44d756e2816f2fad379730874e0266`
- Private prepared binding SHA256:
  `2a132c40552a32dde8f2ce6f009831fa5a69c070515c24c13a1a30834ca26b9c`
- Raw server log SHA256:
  `0773c071118a3fa543936e1a905fcf314b298bd93d2935f79cae2a5b15a051ca`
- Raw supervisor SHA256:
  `8c27a41126c57398cac33ccec72e80993bb73172775fa4c558b331b982694d21`
- Raw result SHA256:
  `3be361036c37cd603352ce8248c11f70774b8d38713336c21782bfd195fc131c`
- Credential-free derived receipt SHA256:
  `fcc477461283ead2f207917bfd8f05af19076172a75fda5cc3bdf0f911c14d15`

`receipts/result.json` contains63 original raw hashes and an allowlisted result
projection, not raw launch credentials, environment, process command lines or
map contents. All private raw files remain under0700 `private-case-20261015-01/`.
These are SHA256 integrity receipts plus maintained launch authentication, not
an invented external digital signature. No Git index/write/commit/push occurred.
