# Numeric process framing: plain CPU measurement and RawValue change

## Scope and provenance

Measured on Intel Xeon Platinum 8369B, Rust 1.99.0-nightly
(ad3d0bc14 2026-07-31), workspace release profile (opt-level 3, fat LTO,
codegen-units 1), no CUDA feature and no IPC instrumentation. The benchmark
asserts these conditions. `Instant` measures warmed CPU-only wall time, not
hardware cycles or scheduler-exclusive CPU time; CPU affinity was not pinned.
There are no pipes, child work, GPU kernels, sleeps or per-byte timers in the
microbench. Do not substitute the historical 279M inclusive copy spans for it.

The existing full-40 F32 CPU oracle supplies actual top-8 IDs, routing weights
and H2048 post-layer hidden vectors. Owner placement is the production
`expert % 8`, not a uniform-routing assumption. Hidden vectors supply realistic
JSON magnitudes/bit patterns, **not the unrecorded expert input/output vectors**.
Reply payload values deliberately reuse those vectors; no expert math is
claimed. These are **not wire captures from the 15-request/480-token run**.

Oracle files under `target/validation/qwen35-fp8-reference-process-20260930`:

- `hello.safetensors`: SHA256 `1b3097a7260975a634aeac222953b0a4a7ff3604e014ffa985813f7946e36db1`
- `capital.safetensors`: SHA256 `a3376689c9fdbca3e7a85984054896dedc5547f4735053a02304a71e2ab45d0e`

All 966 nonempty buckets are measured independently, using the actual
`NumericCommand`, `NumericReply`, `NumericToken`, bounded serializer and process
envelope. Empty buckets are counted but not sent. Each final phase is the median
of five batches of eight calls after four warmups. Owned `from_value` and child
validation input clones are outside their timers. Allocation/destruction caused
by the measured operation stays inside. Artifact reads and fixture setup stay
outside. No production benchmark harness was introduced.

The first pre-change release baseline gave framing-only cumulative totals
172.555/173.700 ms (hello prefill/decode) and 865.291/172.631 ms (capital
prefill/decode). The tables below use the **same final binary's legacy and raw
paths**, removing build-to-build drift; legacy uses the original Value envelope
and payload helper retained only under `cfg(test)`.

## Per-bucket framing results

Bytes are mean command frame body bytes (exclude the unchanged four-byte length
prefix); actual numeric old/new JSON lengths are asserted equal, as are decoded
JSON values. Object member ordering is not byte-for-byte identical. Host bytes
per token per direction are 8192. ns are medians across the observed buckets of
each size. A full roundtrip serializes/decodes/size-checks both directions once;
it excludes numeric domain validation and I/O.

| Routes/bucket | Samples | Command frame B | Before ns | After ns | Reduction |
|---:|---:|---:|---:|---:|---:|
| 1 | 420 | 22797 | 538034 | 461528 | 14.2% |
| 2 | 227 | 45378 | 1067585 | 915158 | 14.3% |
| 3 | 81 | 67890 | 1602911 | 1374389 | 14.3% |
| 4 | 63 | 90451 | 2137101 | 1833586 | 14.2% |
| 5 | 47 | 113085 | 2667954 | 2295419 | 14.0% |
| 6 | 42 | 135615 | 3210486 | 2753709 | 14.2% |
| 7 | 34 | 158215 | 3753124 | 3215517 | 14.3% |
| 8 | 21 | 180737 | 4285877 | 3674341 | 14.3% |
| 9 | 18 | 203170 | 4829928 | 4149739 | 14.1% |
| 10 | 10 | 225822 | 5380725 | 4609723 | 14.3% |
| 11 | 1 | 247339 | 6212829 | 5567129 | 10.4% |
| 12 | 1 | 270785 | 6604026 | 5812650 | 12.0% |
| 13 | 1 | 293241 | 7306551 | 6232057 | 14.7% |

### One-route phase isolation (ns)

The inclusive entries must **not** be added to their children.

| Phase | Before ns | After ns |
|---|---:|---:|
| Typed payload bounded serialize | 25689 | unchanged serializer |
| `payload` inclusive serialize + Value parse | 76310 | serialize + raw validation separately |
| Raw validation/borrow of already serialized bytes | — | 52373 |
| Value envelope serialize (without setup clone) | 32404 | raw bytes copied by bounded envelope writer |
| Envelope decode | 91375 | 53472 |
| Payload size/normalization check | 31818 | 30148 |
| Typed decode (`from_value` / `from_str`) | 43944 | 65856 |
| Reply encode, inclusive | 108858 | 80482 |
| Complete nonoverlapping framing roundtrip | 538034 | 461528 |

## Actual 40-layer x owner distribution and accumulation

Histogram keys are route count per owner bucket, including zero. Each row is
one model call across 40 layers and 8 possible owners. Hello prefill and both
decodes have one input row (320 routes); capital prefill has five (1600 routes).
These sums are a **serial CPU-work model**, not a GPU critical-path measurement.

### capital / decode.0

Histogram: `{'0': 107, '1': 128, '2': 66, '3': 16, '4': 3}`

| Owner slot | Nonempty buckets | Routes | Before ms | After ms |
|---:|---:|---:|---:|---:|
| 0 | 29 | 43 | 23.169 | 19.934 |
| 1 | 29 | 41 | 22.031 | 18.956 |
| 2 | 27 | 50 | 26.835 | 23.055 |
| 3 | 26 | 37 | 19.866 | 17.054 |
| 4 | 22 | 33 | 17.712 | 15.215 |
| 5 | 23 | 33 | 17.735 | 15.251 |
| 6 | 27 | 41 | 22.044 | 18.992 |
| 7 | 30 | 42 | 22.591 | 19.513 |
| **Total** | **213** | **320** | **171.983** | **147.969** |

### capital / prefill

Histogram: `{'0': 5, '1': 18, '2': 29, '3': 39, '4': 54, '5': 47, '6': 42, '7': 34, '8': 21, '9': 18, '10': 10, '11': 1, '12': 1, '13': 1}`

| Owner slot | Nonempty buckets | Routes | Before ms | After ms |
|---:|---:|---:|---:|---:|
| 0 | 40 | 208 | 111.937 | 96.992 |
| 1 | 39 | 197 | 106.144 | 91.883 |
| 2 | 39 | 187 | 100.529 | 87.007 |
| 3 | 40 | 202 | 109.378 | 95.569 |
| 4 | 39 | 214 | 115.293 | 99.490 |
| 5 | 40 | 194 | 104.933 | 90.393 |
| 6 | 39 | 206 | 111.566 | 96.578 |
| 7 | 39 | 192 | 103.960 | 90.012 |
| **Total** | **315** | **1600** | **863.740** | **747.923** |

### hello / decode.0

Histogram: `{'0': 103, '1': 138, '2': 60, '3': 14, '4': 5}`

| Owner slot | Nonempty buckets | Routes | Before ms | After ms |
|---:|---:|---:|---:|---:|
| 0 | 22 | 33 | 17.663 | 15.278 |
| 1 | 33 | 60 | 32.147 | 27.847 |
| 2 | 20 | 25 | 13.441 | 11.602 |
| 3 | 24 | 39 | 20.883 | 18.576 |
| 4 | 29 | 45 | 24.241 | 20.713 |
| 5 | 29 | 37 | 19.958 | 17.576 |
| 6 | 27 | 36 | 19.389 | 16.847 |
| 7 | 33 | 45 | 24.125 | 20.725 |
| **Total** | **217** | **320** | **171.848** | **149.163** |

### hello / prefill

Histogram: `{'0': 99, '1': 136, '2': 72, '3': 12, '4': 1}`

| Owner slot | Nonempty buckets | Routes | Before ms | After ms |
|---:|---:|---:|---:|---:|
| 0 | 33 | 49 | 26.327 | 22.441 |
| 1 | 32 | 48 | 25.639 | 22.327 |
| 2 | 24 | 34 | 18.352 | 15.543 |
| 3 | 25 | 34 | 18.372 | 15.693 |
| 4 | 23 | 35 | 18.667 | 15.954 |
| 5 | 29 | 37 | 19.997 | 16.892 |
| 6 | 26 | 36 | 19.230 | 16.946 |
| 7 | 29 | 47 | 25.093 | 21.450 |
| **Total** | **221** | **320** | **171.677** | **147.245** |

## Numeric domain costs and nonoverlapping model

`numeric_domain_cpu.rs` calls the real private `NumericToken::encode`,
`validate_token_frames`, `validate_tokens`, and `encode_values`, using actual
ExpertTokenBucket values and representative full-40 metadata: 256 experts/layer,
8 owners, 1280 expected sources/owner, H2048, 32-row/256-route capacity. Catalog
hashes/storage sizes are metadata fixtures (not real checkpoint payloads).
Validation reconstructs and verifies `boot.group()` on every call. This has
**not** been cached, removed, or optimized in this change.

| Call | Numeric encode ms | Proxy validate ms | Child validate ms | Numeric reply encode ms | Serial framing + domain before ms | After ms |
|---|---:|---:|---:|---:|---:|---:|
| capital/decode.0 | 0.067 | 151.117 | 151.586 | 0.067 | 474.821 | 450.807 |
| capital/prefill | 0.406 | 226.029 | 227.058 | 0.398 | 1317.631 | 1201.815 |
| hello/decode.0 | 0.069 | 154.023 | 154.060 | 0.071 | 480.071 | 457.386 |
| hello/prefill | 0.069 | 155.680 | 155.953 | 0.074 | 483.453 | 459.021 |

Only disjoint domain operations and the complete framing roundtrip are summed.
The combined serial model exceeds observed per-token process latency, so it
cannot be used as an additive explanation of the historical ~200 ms gap:
metadata fixture costs, cross-owner overlap, placement, caches and different
runtime scheduling matter. No overlap fraction was measured. The reduction of
roughly 23–24 ms in decode **framing CPU work** is not a promised per-token wall
improvement. GPU owner must rerun the same workload.

## Implementation and compatibility

- `runtime/Cargo.toml`: only add `serde_json/raw_value`; no version bump or new
  package. Cargo.lock already had user changes and needed no update here.
- Outbound typed payload is bounded-serialized once, then borrowed as RawValue
  and embedded without a Value tree or second payload stringify. Bound checks
  still precede writer growth. Temporary payload bytes are dropped before I/O.
- Inbound frame owns the prefix-bounded buffer. Its strict message parser borrows
  raw slices in one pass, including tag-last input, and parses metadata explicitly.
  Payload length is checked **before** typed allocation/deserialization.
- Preserve process v1 and numeric v2 schema, integer bit-pattern values, dtype,
  field names, identities and response kinds. No binary/base64 wire, floating
  alias, unchecked memory or skip-verification path.
- Exact JSON object key ordering changes from Value's sorted ordering to typed
  Serialize ordering. Receivers remain order-independent. Raw byte limits also
  charge internal whitespace; noncanonical padded peers may be rejected sooner.
- RawValue alone skips depth and number-range checks. An allocation-free scan
  preserves the full-envelope 128-depth ceiling. Float/large-integer spellings
  retain Value normalization checks so `1e2` or `-0` cannot evade an old canonical
  byte limit. Numeric u32-array frames take no Value fallback.
- Unknown envelope/variant fields and duplicates remain rejected, including
  null duplicates and array-form messages. Direct typed payload decode now also
  rejects duplicate struct fields previously collapsed by intermediate Value
  (strictness increased, never relaxed). Generic Value payload behavior remains
  its normal serde behavior.
- Preserve absolute deadlines before/after decode, pre-send refusal/READY,
  correlation, no replay, Fenced negative CommandDecode, Unknown OutputEncode,
  ManuallyDrop custody, shutdown ACK + reap, quarantine and reaper ownership.
- Existing compile-time observation remains opt-in and inclusive. No new timer,
  runtime flag, counter or per-byte span was added to production paths.

## Validation

- CPU runtime lib: **553 passed**, one ignored microbench.
- Real current-checkout children: process ranks **23**, decoder **14**, network
  ranks **17**, activation adapters **2** passed; three opt-in measurements ignored.
- `cargo clippy --locked -p ferrule-runtime --no-default-features --lib --tests --examples -- -D warnings`: passed.
- Numeric tests preserve finite bit edges/random patterns and compare raw/legacy
  JSON values and numeric frame sizes. Protocol tests cover prefix refusal before
  body allocation, exact raw limits, depth, normalization expansion, unknown and
  duplicate fields, expired decode, negative output-encode responses and custody.
- No GPU execution or CUDA build was performed. No GPU/model source changed.
- No commit was created. Only CPU IPC/numeric tests and runtime manifest were edited.

## Reproduce

From repository root (use absolute paths for env variables, because lib tests
run from the package directory):

```sh
FERRULE_PROCESS_IPC_INSTRUMENT=0 \
FERRULE_NUMERIC_CPU_ORACLE=/home/admin/luhuanbing.lhb/Ferrule/target/validation/qwen35-fp8-reference-process-20260930 \
FERRULE_NUMERIC_CPU_REPORT=/home/admin/luhuanbing.lhb/Ferrule/target/numeric-cpu-repeat.json \
cargo test --locked -p ferrule-runtime --release --no-default-features \
  --lib numeric_process_plain_cpu -- --ignored --nocapture --test-threads=1

python3 crates/ferrule-runtime/tests/bench/summarize_numeric_cpu.py \
  target/numeric-cpu-repeat.json --output target/numeric-cpu-summary-repeat.json
```

First full release link took about eight minutes on this host; benchmark about
254 seconds. Bound build and execution separately when using a job supervisor.
Generated artifacts are intentionally untracked under target:

- `target/numeric-cpu-before.json`: SHA256 `c6445b872aaf958633090bc1b79ee2020275f201c1718bac52572158bf4d256f`
- `target/numeric-cpu-after-final.json`: SHA256 `fd5a59812b4db4976cacb12039c220042722c87fb3353f5ee9cf86d654a9d664`
- `target/numeric-cpu-summary.json`: SHA256 `bc8d6b90881b62efc98ff14bbb6219ed9051f91f971cbee80a0cbcdfe18b75b1`
