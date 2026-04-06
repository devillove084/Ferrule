#!/usr/bin/env bash
set -euo pipefail

cargo test --workspace
cargo test -p ferrule-app --test doctor_mock
cargo test -p ferrule-app --test rollout_mock