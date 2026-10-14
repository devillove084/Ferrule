# Network recovery validation — 2026-09-30

CPU only; no GPU execution, CUDA feature validation, multi-host run, commit or
push. CLI/factory/model and process implementations were not modified by this
network work. The process owner's existing shutdown_until is consumed directly.

## Passed

- `cargo check --locked --no-default-features -p ferrule-runtime`
- `cargo build --locked --no-default-features -p ferrule-runtime --example network_rank_child`
- `cargo test --locked --no-default-features -p ferrule-runtime --test network_ranks -- --test-threads=4`: **17 passed**
- `cargo test --locked --no-default-features -p ferrule-runtime --lib parallel::network -- --test-threads=4`: **9 passed**
- `cargo test --locked --no-default-features -p ferrule-runtime --test process_ranks`: **21 passed**
- Targeted rustfmt check over all new Rust sources.
- `git diff --check` on modified manifest, module export, policy and lockfile.
- Non-strict clippy for runtime lib, network_ranks and network_rank_child completed;
  no warnings located in the new network files. Existing scope-external warnings
  remain.

Network tests used FERRULE_NETWORK_TEST_CHILD pointing to Cargo's current
absolute prebuilt network_rank_child artifact. The test-only content fingerprint
checks the explicit artifact before dispatch and fails closed without compiling
on mismatch. Parallel nested Cargo compilation initially hit the bounded 90s
fixture timeout; prebuilding removes that contention, not by extending deadlines.

The integration suite covers controller + two endpoint **processes** + two rank
children, production DP and TP column/row executors, ragged CPU matrix shards,
outer TP transaction preservation, Unknown/quarantine, no replay ledger, genuine
child Ready, startup hang, exit, idle disconnect and all three shutdown budget
orderings. Unit fault tests use actual authenticated sockets and, for corrupted
or lost execution replies, an actual ProcessRankOwner computation first.

## Dependency policy: not globally green

rustls **0.23.45**, ring **0.17.14**, rustls-webpki **0.103.15**; PEM parsing uses
rustls::pki_types::pem::PemObject, with no rustls-pemfile dependency. Only ISC was
added to deny.toml, with its TLS dependency reason. No advisory ignores added.
Cargo generated the lock normally; the final diff has no existing-package version
upgrades. rcgen and tempfile are test-only dependencies.

`cargo deny --workspace --locked --offline check`:

- advisories: OK (cached DB, not a claim of fresh online verification)
- licenses: OK
- sources: OK
- bans: FAILED solely on the scope-external unversioned path dev-dependency
  `ferrule-model/Cargo.toml:38`, `ferrule-runtime = { path = "../ferrule-runtime",
  default-features = false }`. Its owner should add version = "0.2.0"; do not relax
  wildcards = "deny".

Cached RustSec DB: `f23b768236fe2880e4cfa167da662cad8ca79240`, commit time
2026-09-29T09:32:15+02:00. A later online cargo-deny refresh failed with GitHub
`Empty reply from server`. cargo-deny 0.20.2 was installed with --locked.

Strict `cargo clippy ... -- -D warnings` is blocked by existing
`ferrule-common/src/io_protocol.rs:906` result_large_err (and non-strict output
lists other scope-external warnings). No blanket lint allowances were introduced.
Nextest is not installed, so no nextest result is claimed.

All real test authentication keys are generated under private temporary
TEST-ONLY directories, never in source control. CPU loopback results are not CUDA
fence, multi-host placement, performance or production CLI acceptance evidence.
