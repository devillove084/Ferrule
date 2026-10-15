# Projection, privacy and provenance rules

The immutable raw receipts include sensitive nested contexts. In particular the
historical V2 run has worker credentials (including `original_worker_token`
variants), launch identifiers, environment maps and raw map/mount observations.
These sources are **not safe to publish wholesale**. Neither their previous
publication nor a reported PASS is a license to copy their nested raw fields.

## Explicit public projection

`curate.py` and `curate_bundle.py` define fixed field selections. Every changed
JSON receipt is wrapped as `ferrule.paired-http-publication-projection.v1` with
`redacted_projection: true`, `original_receipt_replaced: false`, the rule, and the
original source locator/byte count/SHA-256. The public member's SHA-256 appears
separately in `manifest.json`. A projection never pretends to have the original
receipt's content hash, digest, authentication capability, or absolute path.
Original protocol/seal digest fields, where retained, describe the **original**;
no private seal is reissued for a redacted copy.

- **Small exact copies:** order, request schedule, final cleanup, per-arm attempt
  and read-ledger assessment, and all 120 dispatch/120 HTTP-terminal reports. The
  checker binds their unmodified bytes to original inventory hashes. Dispatch
  journals are intentionally recorded `censored` before a terminal arrives; they
  are not the final request outcome. Final terminals determine those outcomes.
- **Client/request projections:** numerical timing, public canary text and token
  IDs, request configuration, raw client SSE data and terminal usage, log-window
  bounds, actual token join, all logical invocation rows, and request PID-I/O
  snapshots are retained. Client host URL, arbitrary metadata, interpreter path,
  platform hostname and path-bearing wrapper results are omitted.
- **Server logs:** only the eight explicitly listed request/invocation/shutdown
  event kinds are copied as structured records. Each keeps its original source
  line and byte start/end. Original timestamps and numeric/text values are not
  obfuscated. Non-JSON startup text, listener/network text and unrelated startup
  records are not published. The full original log hash, original length, selected
  event counts and omission rule are recorded; projected log offsets always refer
  to the original log, not compressed/public offsets.
- **Supervision/lifecycle:** recorded result, PID/birth/parent, containment/ECHILD,
  reaping, actual closure, scalar resource observations, idle outcomes/samples and
  deadlines remain. Raw stdout/stderr, handshake, launch IDs, worker authentication,
  complete environments and final process diagnostic contexts are omitted. This
  cannot replay the private authenticated producer or prove its identity anew.
- **Mount counters:** retain selected numeric counters, acquisition windows,
  age/layout, reader PID/birth and original raw hashes. Withhold every device,
  export, mountpoint, mount namespace, boot ID and raw mount line. Equal original
  identity tuples share a sequential `shared-client-mount-1` alias assigned in
  fixed arm/file order. This is an alias, not a fabricated hash or measurement.
- **Maps:** retain the original raw-map SHA-256/size/relative locator, PID/birth
  bracket, actual CLI hash, required library basenames/content hashes, validation
  timestamps, required/observed coverage, and missing/deleted evidence. Raw map
  text, inode/device/overlay/host paths and deleted-file names stay private.
  `live_mapped_memory_sha256` remains null and overlay backing remains unobserved.
- **Device identity:** use stable inventory-position aliases `device_0` through
  `device_7`. Actual physical UUIDs remain private. Equal aliases preserve the
  recorded same-selected-device relationship; they do not recertify hardware.
- **Configuration:** explicit public runtime controls retain log format/filter,
  workspace setting and NVTX off. No arbitrary environment map is published.
  Model/CLI/loopback locations become visibly symbolic roles. Numerical budgets,
  scheduler settings, canary strings, all timestamps and plain timing values stay
  unchanged. Public build controls and exact cargo argument lists are selected,
  not copied from an inherited environment.
- **Paths/pins:** repository inputs use repository-relative locators. External
  artifact locations use an explicitly labeled location alias plus artifact
  basename. The alias is a hash of a private location string, **not a content
  hash**. The untouched original content SHA-256 is separate. No private mount,
  export or network hostname is placed in these locators. A pointer naming a
  binary/source archive is metadata only; the referenced artifact is not copied.
- **Historical failures:** only summaries, all planned arm rows, stop reasons,
  original hashes and separate denominators remain. No historical credentials,
  raw maps, environments or request timings enter the completed-study population.

`provenance.json.gz` inventories every regular file in the selected completed raw
tree, its private plan directory and both failed raw trees, including omitted
files. It retains original global pin sets from the sealed final, audit, plan,
binding, build review, protocol, cases and oracles. Missing external artifacts are
not newly opened or claimed revalidated; their recorded pins are preserved.
Neither source archives nor model payloads are read/copied by the public checker.

## Scanning and scope

The curator scans all inventoried raw JSON recursively for private credential
field variants, including JSON embedded in strings/line contexts. The actual
private literal values are kept only in memory and compared against **every
uncompressed public member**; values are never printed or stored in scan receipts.
`audit_sources.py` repeats this owner-side audit and verifies every raw file hash
unchanged, with no launch, GPU, NAS or network operation.

The offline checker independently rejects credential/environment/private-location
markers recursively, including encoded JSON strings. Negative tests cover nested
`worker_token`, `original_worker_token`, `originalworker_token`, environment,
handshake and nonce contexts. Public model token IDs and the event name
`request_worker_token_enqueued` are not authentication credentials.

No scanner proves the absence of every conceivable secret; the primary control
is fixed field selection, followed by structural and known-private-value scans.
The offline reader has no private literal set and cannot repeat that portion
without the originals. The receipt labels this limitation explicitly.

The manifest hashes all public files except itself (no recursive self-hash).
Validation/privacy receipts record the completed checks at their stated scope;
the final manifest then pins those receipt bytes. Subsequent code/receipt edits
require resealing and revalidation. This release is immutable after commit.
