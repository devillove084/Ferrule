# Network linear ranks: CLI integration contract

The runtime implementation is Unix-only, synchronous, host-staged TLS 1.3.
It owns no model, CUDA context, scheduler, collective or transaction publisher.
CLI wiring is deliberately not part of this patch. This is not decoder/EP/KV
serving, HTTP TLS, NCCL or GPU-direct transport.

## Public API (implemented)

From `ferrule_runtime::parallel::network`:

```rust,ignore
ClientTls::load(&TlsFiles, server_name: &str) -> Result<ClientTls, NetworkError>
ServerTls::load(&TlsFiles) -> Result<ServerTls, NetworkError>
NetworkRankOwner::<Boot, WireCommand, WireOutput>::connect(
    address: SocketAddr, tls: &ClientTls, binding: NetworkBinding,
    config: &Boot, options: NetworkOptions,
) -> Result<NetworkRankOwner<Boot, WireCommand, WireOutput>, NetworkError>
owner.execute(transaction: ExecutionTransactionId, session: u64, &command)
owner.execute_or_quarantine(transaction, session, &command)
owner.shutdown() -> Result<(), NetworkError>
owner.shutdown_until(deadline: Instant) -> Result<(), NetworkError>
owner.shutdown_or_quarantine() -> Result<(), NetworkError>
owner.state() -> NetworkOwnerState // Ready | Unknown | Shutdown
NetworkEndpoint::<Boot>::bind(address, allow_remote_bind) -> Result<TcpListener, NetworkError>
serve_once(endpoint: NetworkEndpoint<Boot>, listener: TcpListener) -> Result<(), NetworkError>
```

`Boot` and `WireCommand` implement Serialize; `WireOutput` implements
DeserializeOwned. Endpoint constructs the actual `ProcessRankOwner<Boot, Value,
Value>` and sends the same bounded JSON to the local child. It never accepts an
executable, argv or environment from the network. `NetworkEndpoint<Boot>` fields:

- `binding: NetworkBinding`, `options: NetworkOptions`, `tls: ServerTls`;
- `launch: ProcessLaunch`, `local_identity: ProcessIdentity`,
  `local_options: ProcessOwnerConfig`, `config: Boot`.

The local identity may differ from the network identity; transaction and session
are preserved exactly. Command sequences are one-to-one, monotonic and never
replayed. Boot is bounded, parsed to canonical JSON and compared against the
endpoint's trusted local Boot **before spawning**. This includes weights/fixture
values. Validate the linear Boot/spec with the existing CLI validator before
constructing the endpoint: the generic transport does not interpret model DTOs.
Do not expose the CLI's general decoder dispatch as a network-selected profile.

## Proposed CLI configuration shape

Add a separate `bench-parallel --remote-manifest` option, mutually exclusive with
local thread/process placement. Do not extend the shared RankBackend enum merely
to enable linear network benchmarks. Add `rank-serve --config` and explicit
`--allow-remote-bind`; default bind stays loopback.

The manifest contains a nonempty `ranks` list. Each entry supplies:

```json
{
  "address": "127.0.0.1:9443",
  "server_name": "rank-node.example.internal",
  "tls": {
    "ca": "/local/tls/ca.pem",
    "certificate_chain": "/local/tls/controller-chain.pem",
    "private_key": "/local/tls/controller.key",
    "peer_leaf": "/local/tls/worker-leaf.pem"
  },
  "binding": {
    "run_id": "unique-run-id",
    "controller_id": "controller-a",
    "node_id": "worker-a",
    "identity": {"epoch": 1, "rank": 0, "owner_instance": 1},
    "target": {
      "mode": "tp_column",
      "world": 2,
      "device": 0,
      "in_features": 7,
      "out_features": 5,
      "max_rows": 4
    }
  },
  "startup_timeout_ms": 30000,
  "command_timeout_ms": 30000,
  "limits": {
    "max_frame_bytes": 8388608,
    "max_config_bytes": 4194304,
    "max_command_bytes": 4194304,
    "max_output_bytes": 4194304,
    "max_error_bytes": 4096
  }
}
```

These JSON DTOs are for the CLI owner to define; `TlsFiles`, `NetworkBinding`,
`LinearTarget` and `ProcessFrameLimits` already derive strict serde parsing.
`NetworkOptions` uses Duration fields, not milliseconds on the public Rust API.
`mode` is `dp`, `tp_column`, or `tp_row`. Validate exact rank coverage, consistent
world/mode/geometry, distinct owner lifetimes and `(node_id, device)` uniqueness.
Different nodes using device ordinal 0 are valid. Use parsed SocketAddr, not DNS.
The endpoint independently supplies the same binding/limits/command timeout and
canonical Boot using its local fixture; it uses server credentials and pins the
controller leaf. Local process ceilings/budgets are additional restrictions.
Do not serialize TlsFiles into the application handshake.

## Existing DP/TP adapters

- DP: `NetworkRankOwner` implements `ReplicaWorker<WireCommand>` directly. An
  existing CLI `RankWorker::Network` wrapper may call `execute_or_quarantine`
  with `request.transaction` and `request.session.0` after normal DTO validation.
- TP: serialize the existing Compute/Apply DTO and pass
  **`request.input.transaction` (TensorWork::transaction)**, NOT the internal DP
  admission ID in `request.transaction`. Preserve global/local rank mapping and
  validate output phase, geometry and transaction using existing CLI checks.
- Use the existing `DataParallelExecutor`/`TensorParallelExecutor`. The TP host
  collective and coordinator alone authorize publication. No second commit RPC.
- Ordinary errors may escape the worker only when
  `NetworkError::is_command_rejection()` is true. `execute_or_quarantine` already
  converts all others to `panic_any(PanicQuiescence::Unknown)`; use
  `shutdown_or_quarantine` for the worker shutdown hook. Never override the panic
  hook with Quiescent simply because TCP closed or an OS PID disappeared.
- A connect error after Hello may mean a remote child started: a CLI factory must
  propagate `NetworkError::Unknown` as the same typed Unknown panic, not demote it
  to an ordinary startup error. There is no automatic connect retry.

## Authentication and operational constraints

Both directions require ordinary CA-chain validation plus exact DER leaf pin.
The TLS server name must match the server certificate SAN. Issue clientAuth and
serverAuth leaf certificates as appropriate; configure valid clocks/expiry and
complete intermediate chains. A pin file contains exactly one leaf, not a chain.
The pin authorizes the locally configured controller/node/run/rank binding;
received identity fields are comparisons, never authorization inputs. TLS
resumption, early data and session tickets are disabled. No plaintext fallback.

Keys must be regular, non-symlink, effective-owner-owned files with no group or
world permissions (0600 recommended). PEM reads are capped at 1 MiB and errors
are redacted; parsing uses rustls' built-in PemObject (PKCS#1/SEC1/PKCS#8).
Protect parent directories and distribute secrets outside version control.
Only file paths are config, never key bytes. Rotate leaf and peer pin together.

One endpoint consumes one listener/connection/lifetime. Do not use TCP readiness
probes: an unauthenticated first connection consumes this one-shot endpoint.
Use local bind-status reporting instead. Remote bind opt-in does not relax TLS;
use firewall and network isolation as additional protection.

All network startup and command work uses absolute deadlines including TLS
buffers, ciphertext flush, partial frame prefix/body, JSON decode and validation.
READY idle has no liveness timeout; after the first readable byte, progress does
not reset the frame budget. Commands are serialized and not multiplexed.
Endpoint process waits are observed against the same frame deadline. On expiry
or socket loss, the existing process observer's unwind cleanup invalidates and
terminates the child; this supplies no device fence.

Shutdown carries only remaining milliseconds, never a cross-host Instant. The
endpoint derives its deadline from **frame start**, capped by its network budget,
then calls `child.shutdown_until(deadline)` (also capped by local process budget).
The controller retains its own absolute deadline throughout; network latency is
not a clock-synchronization guarantee. Only valid local ShutdownAck plus reap
allows a network ACK. Missing/corrupt/lost ACK remains Unknown. TERM/KILL/reap
cleanup has independent finite grace. New connections/epochs do not clear old
GPU custody; externally isolate endpoint/child/device before explicit recovery.

## CPU regression execution

Prebuild `cargo build --locked --no-default-features -p ferrule-runtime --example
network_rank_child`. For cargo test/nextest, set `FERRULE_NETWORK_TEST_CHILD` to
Cargo's reported absolute executable. Its bounded identity probe checks source,
manifest, lockfile, checkout path and feature content fingerprint; a bad explicit
artifact never falls back to Cargo. Without an override, tests build once per
test process with a 90-second bound (prebuild for nextest to avoid lock contention).
All credentials are dynamically generated TEST ONLY identities in private temp
directories. The child computes CPU matrix products and fault ledgers; it is not
a production fallback. Loopback CPU results do not establish CUDA or multi-host
correctness, placement, throughput or device recovery.
