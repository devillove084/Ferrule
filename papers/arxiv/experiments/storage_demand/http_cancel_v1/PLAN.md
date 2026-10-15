# HTTP cancellation / fresh-request recovery v1

One parent-authorized, production-path normal cancellation experiment. No injected
hard error, Unknown outcome, public fault flag, rebuild, profiler, cache flush,
warmup, initial Hello2, retries, or additional GPU job is allowed. This directory
alone is owned. The parent/archivist owns the Git index; this entry never writes it.

## Frozen design (before any GPU launch)

Use the preserved CLI86 (`86c58a017d7c4b53490f536a3b1a133f2f7848cc`), SHA256
`7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01`.
Reuse its existing nine qualification gates, immutable compiled-source archive,
reference producers, fourteen model-file identities and twelve required runtime
library pins through the pressure adapter's execution-role root. Current HEAD is
not a compiled candidate input. No previous qualification is rerun or rewritten.
Maintained profiler-qualified production source is unchanged.

Run one actual full40 Qwen3.5-35B-A3B FP8 Mixed CLI lifecycle, TP1/PP1/EP1,
Lookahead, lazy experts, resident KV. Preserve the exact canonical Lookahead
argv/environment: host64MiB includes metadata8MiB and worker stack2MiB;
64 expert entries, cache320MiB including scratch8MiB; device7GiB, workspace2GiB,
KV16MiB, context128, sequences2, batch/prefill8, cohort1/0. NVTX is **off**;
existing JSON request/token/resource and debug invocation logs remain enabled.
The CPU `prepare` command writes immutable `plan.json` and private input seals
before launch. The separate `run --sole-gpu-owner-granted` consumes this explicit
parent grant exactly once; existing GO prevents replay.

Exactly two possible generative POSTs, serially in the same process:

1. `cancel-qa8`: frozen public prompt `The capital of Japan is`, greedy seed1,
   n1, max_tokens8, streaming with usage requested. Observe the first **nonempty
   SSE content event**, not headers, keepalive, blank frames or empty content.
   Immediately shut down the actual TCP socket and close SSE/connection, before
   any filesystem write, pin validation or resource sample. A content event is
   not asserted to be one model token.
2. Before issuing recovery, passively poll complete raw JSON log lines for at
   most45s. Bind the actual `cmpl-ferrule-N` to request/session N. Require the
   worker runtime terminal `cancelled`, `cancellation_requested=true`,
   `cancellation_submitted=true`, no cancellation failure or restore requirement,
   and matching HTTP `client_disconnect` plus closed-channel cancelled terminal
   enqueue. Neither socket close nor a cancellation flag proves runtime acceptance.
   A natural `finished`/max_tokens race stops the experiment; recovery stays
   unissued. No extra inference, tokenize, active state-progress or health request
   is used. Maintained GET `/v1/models` startup readiness is allowed.
3. `recovery-qa8`: only after acceptance, issue one **new** HTTP request/session,
   same frozen QA8, and read through exact text, usage, length terminal and DONE.
   Require actual worker-enqueue IDs
   `[25358,13,198,760,6511,314,6124,369]`, text
   ` Tokyo.\nThe capital of Japan is`, usage5/8/13, unique ordinal1..8, matched
   HTTP/worker terminal and no extra admissions/tokens. No retokenization.

These are two generative requests; step2 is observation, not another request.
Cancel is a **controlled censor**, never a quality success or a missing-usage
failure. Record the actual worker enqueue prefix/count, which may exceed one;
client consumed content and worker enqueues are separate boundaries. Canceled
client usage/DONE remain null. Worker-generated counts may differ from enqueues.
Expected-outcome acceptance is distinct from quality/metric eligibility.

## Timing, custody and evidence

One non-sliding lifecycle envelope: startup60s, at most120s of HTTP,45s shared
cancellation/control/validation, and45s cleanup, total270s (under300s). Request
phase absolute cutoff225s includes pin hashing, maps and observers; each HTTP
operation is capped60s and terminal wait45s (also bounded by its request cutoff).
Timeouts stop, never enlarge a phase or authorize a replacement. Cleanup offsets:
shutdown245s, worker249s, supervisor cleanup260s, post-idle265s, final270s.

Reuse maintained serial journaling, authenticated launch guard, birth/descendant
verification, root capture, original Popen direct-child shutdown, owned subreaper,
all-eight UUID locks, raw-map capture-before-validation, source pins, resource
monitor, NFS/PID-I/O observations and V2 three-consecutive-sample stable idle.
No signal to foreign/numeric grandchild PIDs. Known anonymous deleted mappings
and explicitly labeled overlay backing ambiguity follow the maintained map policy;
all twelve pinned role libraries are required at Ready and after requests.

Use actual per-invocation serial log-window counters/cache observations before
and after requests and final owner shutdown summary. These windows are **not**
transaction/request identity joins. Missing per-request ledger, pending, epoch,
KV/frontier snapshots remain null, not invented zeroes. Record final owner-local
known close/retained residency separately from ECHILD, absent owners, all-eight
idle and lock release/reacquisition; none proves universal physical release.

Successful recovery supports only new-generation/session reuse with the frozen
continuation and no stale continuation **observed in these eight tokens**. It is
not same-request resume, arbitrary KV equivalence, Unknown recovery, an integrated
fault ledger, or a general fault-tolerance result. Cooperative cancellation cannot
interrupt a synchronous read; it must wait for genuine backend progress.

All raw launch credentials, environment and maps remain under an exclusive0700
private directory with restricted files, whole raw inputs/hashes preserved. Only
an allowlisted derived receipt and source/plan/tests are publishable. No raw
launch/supervisor/observer/proc-environment records may be staged.
