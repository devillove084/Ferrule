# Storage-demand repeated full-QA pilot v2

The additive CPU-preparation protocol is in
[`papers/arxiv/experiments/storage_demand/newversionedprotocol/README.md`](../papers/arxiv/experiments/storage_demand/newversionedprotocol/README.md), with the sealed preregistration in `protocol.json` and the narrow adapters in `runnerhelpers/`.

This is not a production or manuscript change. It does not alter or execute the
historical `storage_demand/run_middle.py`, `case-middle.json`, `cpu-revisions/`,
or `runs/` entries. It also does not add a general-purpose harness.

## Decision record

- The old four-request case is not silently extended. v2 raises the per-lifecycle
  ceiling from 10 to 12: `4 initial + 1 excluded-first QA + 2 warmups + 5
  measured QA = 12`.
- There are five independent server lifecycles as a finite pilot budget only.
  This is not a statistical-significance, power, confidence-interval, or
  steady-state claim.
- The five-lifecycle order is fixed, with no retries or replacement trials.
  Scheduled, issued, successful, failed, censored, and unissued denominators
  remain visible.
- A historical n=1 `-02` QA E2E of `18.309471419081092` seconds is a planning
  source only. All 60 requests may cost approximately `1098.5682851448655`
  seconds; the conservative planning allowance is about one hour and the
  literal finite worst-case pilot envelope is two hours (`5 * 1440s = 7200s`).
- Physical NFS/NAS I/O remains scoped as unknown: passive client mountstats are
  timestamped and raw, but shared-client counters are not server-attributable,
  PID `read_bytes` is not NFS traffic, and NAS backend disk bytes are not
  observed.
- Worker token enqueue IDs/ordinals, actual HTTP completion IDs, and client
  content consumption are separate joins. Opaque IDs, gaps, duplicate bindings,
  and missing traces are not repaired by guessing.
- `resource_shutdown_summary` is retained as owner-local evidence. Retained
  residency is not itself unknown custody; explicit unknown scope/fence/closure,
  pending work, quarantine, or lost PID/birth/parent/UUID custody stops. Missing
  final release coverage is `unverified`, not fabricated `verified`.
- UUID locks are local nonblocking serialization, not global reservations. No
  foreign PID is killed, no cache is dropped, and no storage-cold claim is made.

## Gate before any future owner-run

A new immutable current release binary with a matching `StorageMiddle64` effective
configuration must exist first. The GPU owner must then supply fresh, same-source
full40, retained fresh-quality, and cold-owner fresh-quality PASS receipts with
actual generated IDs, independent-reference pins, authenticated custody, and a
pinned compiled-source relation to that exact CLI binary. CPU preparation can
validate and seal those receipts; it never launches the binary or queries a GPU.

Run only the offline checks from the protocol README. There is deliberately no
execution command in this handoff.
