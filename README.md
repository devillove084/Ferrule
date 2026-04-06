# Ferrule

Ferrule is a Rust-native runtime and training scaffold for **agentic reinforcement learning** on top of **Candle**.

The project is intentionally opinionated:

- **runtime first**, not trainer first
- **trajectory as an event log**, not just final text
- **tools and environments as first-class citizens**
- **observability from day one**: structured logging, metrics, and clear startup wiring


## Workspace Layout

```text
ferrule/
├── Cargo.toml
├── rust-toolchain.toml
├── README.md
├── configs/
│   ├── rollout.toml
│   └── train.toml
└── crates/
    ├── ferrule-core/
    ├── ferrule-candle/
    ├── ferrule-runtime/
    └── ferrule-app/
```

Quick Start
1. Check the workspace
cargo fmt
cargo check
2. Run the doctor command
cargo run -p ferrule-app -- doctor --config configs/rollout.toml

This validates the config path, initializes observability, and runs a tiny Candle sanity check.

3. Run a mock rollout
cargo run -p ferrule-app -- rollout --config configs/rollout.toml

You should see a JSON trajectory printed to stdout.

Observability

Ferrule initializes observability as early as possible.

Logging

Logging uses tracing and tracing-subscriber.

Set log level with:

RUST_LOG=debug cargo run -p ferrule-app -- rollout --config configs/rollout.toml
Metrics

Prometheus metrics are exported on the configured address:

[observability]
metrics_enabled = true
metrics_bind = "127.0.0.1:9000"

You can inspect them with:

curl http://127.0.0.1:9000