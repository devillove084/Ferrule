# Claim–evidence–limit ledger

**Ferrule: Bounded Storage-Backed Weight Residency**

This ledger separates intended contracts, source-qualified tests, actual serving
observations and unobserved physical claims. It is a claim map, not an experiment
chronology or a second metrics table. The [artifact guide](ARTIFACT_GUIDE.md)
contains result labels, public inputs, commands, source identities and external
raw-evidence locators. Historical receipts retain their original outcomes.

## Claims and evidence

| Claim | Evidence | Boundary |
|---|---|---|
| Source-bound, exactly matched operations and retained custody provide the intended safety discipline | Design algorithms, exact-key/lease implementation and targeted fixtures; [source crosswalk](ARTIFACT_GUIDE.md#event-and-source-crosswalk) | Conditional invariant/implementation obligation, not formal verification, durable ACID, universal registry coverage or eventual reclamation |
| A selected quality-checked model profile completes within configured logical accounting policies | Earlier factory qualification and **paired-study**; separate **grouped-quality**, **http-transfer-acceptance** and **http-transfer-pilot** on their own source cuts | One Qwen3.5 BF16/F32 execution profile, not full-model W8A16, a total physical-memory bound, another-model qualification or proof that compact weights cannot be resident; newer raw inputs remain local/external |
| Coupled cache limits change retained occupancy without reducing the observed logical read work | **cache-pressure**, three completed blocking points | Fixed order, one lifecycle per point, coupled entry/byte caps; no observed expert hits or read reduction, independent LRU experiment, working-set-fit threshold or capacity frontier |
| Application demand and shared-client observations can be reconciled | **paired-study** and **cache-pressure** request/invocation/owner counters and shared NFS-client snapshots | Logical reader/H2D bytes are not physical storage/bus traffic; backend attribution and missing phase PID-I/O remain unknown |
| Same-budget lookahead is associated with lower observed wait and client latency in the earlier read-scheduling study | **paired-study**, five qualified blocks under its frozen implementation and schedule | Read-schedule comparison only; not the newer grouped/synchronous transfer-policy comparison. Requests are not independent replicates; no causal decomposition, general throughput, guaranteed CI coverage, significance or performance leadership |
| A persistent reader reuses one worker across predictions while retaining idle ownership | **persistent-worker**, nine actual GPU gates; one creation and 159/2,559 reuses at selected full-forward/fresh endpoints | Live idle charge 10 MiB and zero live joins are observations, not zero host residency; final-owner join remains unobserved. Do not overwrite earlier per-forward worker evidence with later implementation counters |
| Grouped synchronous uploads preserve scoped numerical/path checks | **grouped-quality**, 14 actual GPU gates: backend 1, model 5, composition 8 | Own source and executable identities; backend six-copy/one-sync assertion is not a directly observed model-wide grouped-call count or a new serving-performance result |
| Grouped uploads have lower observed whole-test time in completed pairs | **grouped-whole-test**, eight pairs, four per read schedule | Approximately 3.9% median reduction is descriptive and includes construction/IO/checks/teardown, not request/TTFT/token latency. The containing 24-job stage fails at its first profile after 20 passes; no 24/24 claim |
| Grouping reduces observed selected-process native stream-sync rows at equal observed H2D/kernel work | **grouped-api-observations**, qualified within-schedule pairs for blocking and lookahead; separate collection-control cuts | 382,743 → 257,163 rows, −5 per 25,116 logical uploads. Partial selected-process summaries only; full trace false, drops null, exhaustiveness and expert-range/causal attribution unproved. Failed historical captures stay failed |
| Grouped transfer has conditional serving benefit rather than a universal improvement | **http-transfer-acceptance**, four actual servers / 48 checked requests; separate **http-transfer-pilot**, eight servers / 96 requests, 40 measured QA8 / 320 measured tokens, two pairs per schedule | Lookahead TTFT/E2E lower and phase goodput higher; blocking TTFT regresses while E2E is nearly flat. Acceptance timing aggregates null. Client-first-content is not first GPU token; no per-request independence, CI/significance or external ranking. The frozen pilot did not benchmark the post-cleanup resolver: ordinary defaults are blocking/synchronous; omitted transfer policy selects grouped only for supported explicit bounded lookahead. Selector validation is CPU-only, not new GPU qualification |
| A retained-upload alternative is slower in the observed whole-test population | Eight negative unprofiled pairs in **retained-upload negative result**; separate partial profiles show substantial pin/free host API sums | Negative timing remains negative; API sums overlap and are not an exact causal decomposition. The rejected production path is removed by cleanup; historical reproduction uses pinned Git, not a retained current production flag |
| Selected source-read envelopes intersect actual expert-kernel activity | **scoped-overlap**, two accepted profiled QA executions with unique launch correlations | Scoped acceptance only; full trace rejected, three wrapper warnings per arm retained, event loss unknown; userspace pread wall time is not active NAS service or a profiler-free timing comparison |
| Clean abort and sticky Unknown custody are exercised at a real Mixed CUDA component boundary | **mixed-fault**, portable actual records for two selectors / four cells; serialized snapshots and explicitly typed assertion evidence | Tiny deterministic fixture, not the serving binary or 35B fault test; Unknown was synthetic proof loss after synchronization; no runtime broker or full-model numerical oracle |
| Ordinary client disconnect can be followed by correct fresh-request service | **http-cancel**, one lifecycle with controlled cancellation and exact recovery | Cancelled request is censored, not an eight-token quality success; recovery is a new session, not same-request resume, Unknown recovery or interrupted synchronous read proof |
| Normal owners reach recorded closed/retained and process-drained states | Serving closure records, authenticated ECHILD, idle and lock reprobes | Not universal post-drop physical release or a device fence; private authentication is not independently replayed by public receipt checkers |
| A later visible platform inventory is linked to the paired-study boot | **platform-inventory (post hoc)**, CPU model/logical-processor/MemTotal observations and 300 matching phase snapshots across ten arms | Later observation under a no-hotplug/reconfiguration assumption; not installed RAM, physical cores, experiment-time affinity/cgroup allocation or binding to other studies |

## Formal and accounting scope

The credit-conservation equation is an **intended invariant** under explicit
ownership and transfer rules. It is not an experimentally proved theorem about
all deployed paths. The resource broker exposes capacity, execution reserve,
in-use and high-water accounting. No complete five-bucket transaction ledger was
collected for the serving experiments, and the Mixed fault fixture has no runtime
broker. Do not replace missing broker observations with cache statistics.

The actual scopes are distinct:

- The 64 MiB logical host reservation accounts for bounded read slots, metadata,
  requested worker stack and retained payload aliases. It is not total process
  RSS/PSS or a physical host-memory bound.
- Cache charge accounts for resident, pending-upload and scratch bytes. Unknown
  or quarantined bytes are a subset of charged bytes, not another additive pool.
- Sampled device-wide memory is neither an exact continuous peak nor an exact
  process-owned allocation. Paired/pressure evidence contains 515/108 samples;
  the observer waits two seconds after collection rather than guaranteeing a
  fixed start-to-start period. Independently timed maxima must not be added.
- Physical conservation needs disjoint allocation domains and justified release
  evidence. Zero logical charge, an idle device or process exit does not supply
  that proof. Unknown must remain owned/charged rather than silently refunded.

Persistent worker lifetime adds a distinct retained-host state: an idle, live
owner can legitimately keep metadata and stack charges without keeping a
payload or joining the worker. Zero pending/Unknown/upload charges at selected
endpoints establish those logical observations only. They do not establish an
integrated five-class ledger, continuous alias/refund tracking, RSS/memlock
bounds, final-owner join, or universal CUDA release.

Caller cancellation also does not revoke an already-started irreversible
publication action. A source file implementing a common registry contract does
not establish that every specialized expert, HostKV or prepared-KV path uses it.

## Post-hoc platform interpretation

A later inventory records an Intel Xeon Platinum 8369B model string, 128 visible
logical processors and 754.913 GiB kernel-visible `MemTotal`. Acquisition occurred
at 14:13:09 UTC on 15 October 2026, after the paired study at 00:48–01:13 UTC.
Matching boot hashes in 300 original NFS boundary snapshots support association
with those ten arms, not proof that hardware or resource controls were unchanged.
The public receipt retains the provenance relationship without exposing raw boot
identifiers; offline checking does not reauthenticate the private originals.

Installed physical RAM, physical topology, experiment-time affinity/cgroup
limits and hidden ancestor limits remain unknown. The relationship must not be
transferred to pressure, profiling or other experiment groups. Neither the large
visible-memory inventory nor a small configured staging budget proves the
capacity available to a past process. Physical RAM insufficiency, compact-weight
residency impossibility and physical NFS/storage necessity are not demonstrated.
See the [full scope](ARTIFACT_GUIDE.md#post-hoc-platform-inventory).

## Qualification and statistical interpretation

The two earlier nine-gate factory groups, the later persistent-worker nine,
the retained-upload 1/11 backend/integration stages, the grouped 14 gates, and
the public-CLI acceptance cells are separate source cuts and test roles, not
replications of a single qualification or a pooled quality denominator. Constructor, full-model diagnostic,
retained-fresh, ordinary and cold-owner paths retain their own denominators.
Ordinary route/state observations are null. Source equality and shared runtime
pins do not make a test executable mathematically equivalent to the serving CLI.

Factory tolerance checks are distinct from HTTP output qualification. Worker
IDs are actual successful enqueue observations, not client API token IDs.
Positive HTTP completion/usage/SSE joins establish delivery where observed;
a controlled disconnect intentionally lacks that completion. Smoke requests,
oracle requests and a censored prefix are not interchangeable quality evidence.

The paired estimator uses five paired blocks. Its enumerated resampling ranges
are descriptive, not coverage-guaranteed 95% confidence intervals. A separate
sum-of-request-window token rate is not the paired estimator, startup-inclusive
throughput or concurrent-server capacity. Fresh server owners do not imply cold
storage caches or independent shared-client conditions.

The newer HTTP pilot compares grouped/synchronous transfer within each read
schedule, using two paired blocks per schedule and no schedule pooling.
Cross-block independence is not established under shared cache state. Its
phase-qualified goodput divides qualified measured tokens by the interval from
first measured dispatch to last measured terminal, including inter-request
gaps. The request-only serialized-service rate is different. The earlier
five-pair study compares read schedules and uses a different serial-window
metric; its 8.18%/11.24%/10.91% summaries cannot be attributed to upload grouping.

The two zero-spawn transfer pilots retain all scheduled requests as unissued
and all timing ratios as null. The later reviewed role-partitioned admission
controls establish successful preparation within unchanged budgets, not faster
model execution. One-time full build identity and fresh live-role validation
have distinct roles; neither replaces runtime/model quality evidence.

The overlap study is separate: read-envelope/kernel and pread-wall/kernel
intersections are unions of actual intervals on one collector clock. Kernel
gaps are not filled, and CPU submission spans are not kernel activity. OSRT
return encodings do not provide per-call byte counts; source-verified operation
bytes cannot be relabeled OSRT or physical NAS bytes. Scoped acceptance does not
repair the rejected full trace or upgrade older rejected captures.

## Fault and recovery interpretation

The clean component branch checks committed GDN/KV bit equality and transient
cleanup; its final drop/trim assertion returns the fixture allocator to its
baseline. That is narrower than device-wide or serving-owner release. The Unknown
branch discards consumer proof only after a successful real wait/synchronization,
retains owners, and rejects reuse even after later independent synchronization.
It is not a physically lost driver proof, forced hang or successful Unknown
reclamation. Post-Unknown state-content equality and physical release remain
unobserved. The portable archive distinguishes serialized boundary/snapshot
observations, the serialized clean-content assertion result (not raw state bits),
and source-plus-actions-plus-PASS evidence for later independent waits, rejected
APIs and clean tx3. Those later facts are not separate per-call result receipts;
only execution and abort assert typed `CompletionUnknown`, while other rejected
APIs assert errors without that stronger type claim.

The HTTP cancellation result adds a production-path observation, but not the
complete integrated fault matrix. Per-request pending-operation, KV, frontier
and epoch snapshots remain null; final owner counters do not backfill them.
A fresh recovery prefix does not establish arbitrary KV equivalence or
same-request continuation. Request cancellation is not evidence that an
individual checkpoint read was interrupted.

Integrated stale/delayed completion, cancellation at every stage, cleanup failure,
owner loss, lost ACKs, duplicate/original submission, retained-credit accounting
and publication/reuse obligations remain open. These are not made complete by
combining unrelated component and HTTP denominators.

## Availability and exclusions

Public qualification, paired, pressure, scoped-overlap and mixed-fault archives
support CPU receipt checking; the HTTP projection supports repository-based
receipt tests. The fault archive and its figure inputs are portable without
private originals, but native binaries, full source archives and redacted custody
originals remain external. Raw qualification for the earlier serving source still has
only a public review/identity projection, not a complete portable raw corpus.
The public paired renderer verifies all 623 archive files and reconstructs all
three figures without private raw inputs. Its byte identity is tied to current
public provenance and the recorded renderer/dependency/font pins. The legacy
private derivation is not a public-reproduction prerequisite. The newer worker, grouped-quality, whole-test, profile and HTTP-transfer
summaries are bounded derived projections with curated-file manifests and
local-original hashes. Their raw reports/traces/SQLite, binaries and native/
model/build inputs remain local or external, not complete portable public
archives. Manifest verification supplies neither original authentication nor
independent raw-metric reconstruction. Source pins keep the earlier serving
CLI, later measured CLI, experiment executables, and collection controls
separate; cleanup/current HEAD is not newly built or GPU-qualified by this
paper integration. Historical rejected flags/selectors require their pinned
Git revision, not restoration of dead production code. The separate platform
supplement supports offline receipt tests, not reconstruction of experiment-
time hardware measurements.
See [reproduction levels](ARTIFACT_GUIDE.md#availability-and-trust-boundary).

Earlier paired stops, pressure stops/admission rejection, rejected profiler
captures, failed numerical/wrapper gates and unmatched resident-TP2/vLLM
observations retain their original denominators and limitations. They are
indexed in [historical exclusions](ARTIFACT_GUIDE.md#historical-exclusions-and-provenance),
not pooled with the current scientific results. Later idle, a corrected
collector predicate or a successful independent generation never repairs an
original failed outcome.

The pinned related-work audit establishes that FlexGen/ZeRO-Inference describe
disk/NVMe weights and SGLang's inspected specialized expert-pack path implements
runtime storage demand. It does not qualify that upstream path's release,
execution or quality, and its restricted models/parallelism are not a universal
claim about SGLang. Inspected vLLM host-weight/UVA offload is distinct from
storage-backed weights and from KV offload. Unmatched resident-system numbers
remain in the artifact guide only, with no winner inferred. DeepSeek-V4 is
untested in Ferrule end to end: architecture capability is not an adapter
performance or model-generality result.

Broader physical attribution, independent repetitions, instrument-cost studies,
fair external baselines, long/concurrent workloads, other precisions/topologies
and alternate modalities remain future work. The
[completion checklist](PAPER_COMPLETION_CHECKLIST.md) tracks those gates separately
from final manuscript, citation and author-declaration review.
