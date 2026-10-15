# Post-hoc platform inventory v1

This additive, CPU-only supplement records **current visible resources**, not a
retrospective measurement of per-experiment resource limits. It neither changes
an experiment protocol nor modifies the manuscript or any existing archive.
No GPU, driver, NVML, model, network or Git command was executed by the collector.

## Recorded result and methods wording

Receipt: [`posthoc-20261015.json`](posthoc-20261015.json).
Status: **`POSTHOC_INVENTORY_SAME_BOOT`** for the completed paired-read
`artifacts-v3-mapfixed-20261015-01` study only.

Suggested methods sentence (keep the qualifications):

> A passive post-hoc inventory on 15 October 2026 at 14:13:09 UTC recorded an
> Intel Xeon Platinum 8369B CPU model string, 128 visible logical processors,
> and 754.913 GiB of kernel-visible memory (`MemTotal`); all 300 original NFS
> phase snapshots across the ten paired-read lifecycles share the inventory's
> boot-ID hash, supporting this platform description only under a no-hotplug
> or reconfiguration assumption, while installed RAM, physical-host topology,
> experiment-time affinity and cgroup limits, and hidden parent-group limits
> remain unknown.

The study's recorded interval is **2026-10-15 00:48:28.013551438 through
01:13:47.738082277 UTC**. CPU/memory/resource acquisition was
**14:13:09.868869046 through 14:13:09.874909108 UTC**, with monotonic brackets
**9016532905407245 through 9016532911443154 ns**. The encompassing provenance
check and collection ran from 14:13:07.944684588 through 14:13:09.874965482 UTC.
These are post-hoc timestamps, not backdated study observations.

| Field | Actual value / source | Scope and limitation |
|---|---|---|
| CPU model | `Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz`, `/proc/cpuinfo:model name` | Present in all 128 unique processor records; frequency is model-string text, not a measured clock |
| Logical processors | 128 distinct `processor` IDs | Visible proc view, not CPU quota, reserved cores or an experiment allocation |
| Socket IDs | 1 distinct `physical id` (0) | Visible topology only; **not** a one-socket physical-host claim |
| Core IDs | 128 distinct `(physical id, core id)` pairs | Derived from actual records, not a CPU-spec lookup; virtualization/physical-host topology is not established |
| Online CPUs | `0-127`, `/sys/devices/system/cpu/online` | Current online mask |
| Collector CPU affinity | `0-127`, `/proc/self/status:Cpus_allowed_list` | Current collector only, not each experiment's mask |
| NUMA view | `node0`, CPUs `0-127`; collector `Mems_allowed_list=0` | Current visible sysfs/status view, not physical NUMA attestation |
| Kernel-visible memory | `MemTotal: 791584008 kB` | Linux proc `kB` means 1024 bytes: **810582024192 bytes = 754.9133377075195 GiB = 810.582024192 GB** |
| Installed physical RAM | **Unknown** | No DIMM, SMBIOS, serial-number or machine-inventory probe |
| Kernel / architecture | Linux `4.19.91-009.ali4000.alios7.x86_64` / `x86_64` | Current `uname` release/machine only; hostname and build-node fields omitted |
| Current cgroup memory limit | v1 `memory.limit_in_bytes=9223372036854771712` | Unlimited sentinel at the read scope, **not** available memory or installed RAM |
| Current memory+swap limit | `memory.memsw.limit_in_bytes=9223372036854771712` | Same sentinel; `memory.use_hierarchy=1`; the two `memory.stat` hierarchical-limit fields also have this value |
| Current CPU quota | `cpu.cfs_quota_us=-1`, `cpu.cfs_period_us=100000` | No finite local quota; does not mean 128 CPUs were allocated to an experiment |
| Current cgroup cpuset | `cpuset.cpus=0-127`, `cpuset.mems=0`; effective files match | Collector's current group only |
| Parent cgroup limits | **Unknown** | Current memberships resolve exactly to non-root, mounted subtree roots; ancestors above those roots are not visible |
| Experiment-time CPU/RAM/affinity/limits | **Not recorded in the inspected study metadata** | Same boot is not evidence that these properties remained unchanged |

Use `GiB = bytes / 1024^3` and `GB = bytes / 1000^3`. Do not relabel this as
“768 GB installed RAM,” “128 physical cores,” or “one physical socket.”

## Original evidence and binding

The collector inspects the originals **before** collecting current resources:

1. Reads the existing public `publication_results_v1/manifest.json` and
   `provenance.json.gz`, without importing experiment runners or opening any
   referenced model/library/source-archive payload.
2. Inspects all **598 public JSON/JSON-gzip files** (including the manifest),
   checking the other **597** against its compressed/raw size and hash pins.
3. Checks all **1,037 original JSON files** in the completed raw study against
   the public provenance's original byte counts and SHA-256 pins.
4. Reconciles each arm's original/public NFS phase set, all original source
   pins, capture clocks/windows and raw-text hashes. Checks all **300 original
   mount-text files** against both their original JSON and public provenance.
5. Finds **341 boot-ID occurrences**, all matching the current boot hash:
   300 phase snapshots, 10 custody files, 10 supervisor files, 10 assessment
   files, 10 nested final-result occurrences, and one registration occurrence.
   Repeated nested copies are **not independent host measurements**.
6. Counts **30 NFS phase snapshots per arm, all matching**, for `P1` through
   `P5`, each with blocking and lookahead. All 30 phase labels and per-arm time
   ranges are preserved in the receipt. These NFS provenance counts are not
   GPU-observer sample counts.
7. Searches a recorded fixed set of CPU/memory/kernel/affinity/cgroup JSON keys;
   there are no hits in this scoped original/public metadata. An additional
   inspection of original JSON/text/log sources found no `MemTotal`, affinity,
   resource-limit-file values, CPU model, or recorded uname command. Existing
   mount metadata establishes mounted cgroup filesystems, not their limits.

The current and every captured study boot ID yield:

```text
SHA256(stripped original UUID text, UTF-8, without newline)
26168601009363e17137df185a3070b224013329ce1a300831506e8e72a9d080
```

The raw boot UUID is read in memory and **never copied to the public receipt**.
Originals remain private and unchanged. Public source pins let an owner repeat
the comparison against originals; they do not let an offline reader recover
or independently verify withheld boot IDs. Hashes are not signatures. The
collector also checks that the current boot hash did not change during its run.
Missing/invalid/mismatched boot IDs block study binding; no current hardware
inventory is collected after a failed initial binding. A changed boot during
collection invalidates the result. There is no replacement study or retry of
GPU work.

**Do not extrapolate this binding to pressure, profiling, reference-generation,
or other experiment groups.** Those groups were not boot-bound by this receipt.
Existing GPU/model/software facts must retain their existing recorded sources.
The frozen reference manifests in
`../storage_demand/qualification_v1/receiptarchive/references/` record
`torch=2.13.0+cu129` and `transformers=5.5.3`; these are historical reference
versions, not a software probe or confirmation of current installed versions.
No GPU inventory is newly collected or certified here.

## Safe figure/table facts

For an architecture/setup schematic, use:

- **CPU: 128 visible logical processors (post hoc)**, with the Xeon model label.
- **Visible memory: 754.913 GiB MemTotal (post hoc)**, not installed RAM or an
  expert-cache allocation.
- **One visible NUMA node; collector affinity 0-127 (post hoc)** only if relevant.
- A provenance note: **same boot hash, 300/300 phase snapshots, 10/10 paired-read
  arms**. This is a provenance relationship, not a performance result.

Keep topology counts in a qualified table rather than drawing a physical
one-socket/128-core machine. Do not turn unlimited local cgroup sentinels into
capacity bars. Hidden ancestors and experiment-time limits remain unknown.
GPU labels and all observer/pressure denominators belong to existing artifacts,
not this new inventory.

## Files, safety and validation

- `inventory.py`: Python 3.10+ standard library only, explicit passive allowlist.
- `posthoc-20261015.json`: small public post-hoc receipt; selected values and
  hashes only, no full proc files or original snapshot copies.
- `test_inventory.py`: synthetic CPU-only parser/binding fixtures and offline
  receipt consistency/privacy tests; no live GPU or hardware reads.
- `README.md`: this interpretation and reproducibility note.

Cgroup resolution uses `/proc/self/cgroup` **and** `/proc/self/mountinfo`, subtracts
an exact matching mount root from the current membership path, and reads only
memory/CPU/cpuset control files along that visible path and its visible ancestors.
It does not assume `/sys/fs/cgroup/<controller>/` is the collector's group, probe
unrelated container bind mounts, or infer hidden parent settings. Membership
and mount paths are hashed, not published. v1 values `-1` or at least `2^60`
are classified as unlimited sentinels; the actual captured value is
`2^63 - 4096`. The raw value and threshold remain explicit. v2 uses literal
`max`; numerical v2 values remain numerical limits. CPU quota is a CPU-time
ratio, not a topology count.

No hostnames, machine IDs, MACs, serials, IP addresses, NAS export identities,
credentials, raw boot UUIDs or arbitrary environment maps are selected. No
subprocess, package, GPU/driver/NVML initialization, model load or network call
is used. No existing archive, manuscript, production file or Git state is
changed.

Run offline tests from the repository root:

```sh
python3 -S -B papers/arxiv/experiments/platform_inventory_v1/test_inventory.py
```

To collect a **new** timestamped receipt on an authorized Linux host, choose an
unused output name inside this directory. It needs the private originals and
adjacent public paired-results archive. The existing receipt is never overwritten:

```sh
python3 -S -B papers/arxiv/experiments/platform_inventory_v1/inventory.py --output papers/arxiv/experiments/platform_inventory_v1/posthoc-new.json
```

This command passively reads proc/sysfs/cgroup files; it does not reproduce the
experiment. If boot binding fails it emits a `cannot_bind_posthoc_to_study`
receipt with a null current inventory and exits with status 2. Validation/read
failures stop with status 2 without emitting potentially private exception text.
