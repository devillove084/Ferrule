#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-${FERRULE_E2E_MODEL_DIR:-}}"

if [[ -z "${MODEL_DIR}" ]]; then
  echo "usage: scripts/e2e_real_llama.sh <local_model_dir>"
  echo "or set FERRULE_E2E_MODEL_DIR"
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "error: model dir does not exist: ${MODEL_DIR}"
  exit 1
fi

MODEL_DIR="$(cd "${MODEL_DIR}" && pwd)"
export FERRULE_E2E_MODEL_DIR="${MODEL_DIR}"

FEATURES=""
UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"

if [[ "${UNAME_S}" == "Darwin" && "${UNAME_M}" == "arm64" ]]; then
  FEATURES="metal"
elif command -v nvidia-smi >/dev/null 2>&1; then
  FEATURES="cuda"
fi

echo "Using model dir: ${FERRULE_E2E_MODEL_DIR}"
echo "Using features: ${FEATURES:-<none>}"

if [[ -n "${FEATURES}" ]]; then
  cargo test --release -p ferrule-app --features "${FEATURES}" --test e2e_real_llama -- --ignored --nocapture
  cargo test --release -p ferrule-app --features "${FEATURES}" --test e2e_real_generate -- --ignored --nocapture
else
  cargo test --release -p ferrule-app --test e2e_real_llama -- --ignored --nocapture
  cargo test --release -p ferrule-app --test e2e_real_generate -- --ignored --nocapture
fi