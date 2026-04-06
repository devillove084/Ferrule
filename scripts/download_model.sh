#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/download_model.sh <repo_id> <local_dir> [revision]

Examples:
  scripts/download_model.sh HuggingFaceTB/SmolLM2-135M-Instruct ./models/SmolLM2-135M-Instruct
  scripts/download_model.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 ./models/TinyLlama main

Optional env:
  HF_TOKEN=hf_xxx
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

REPO_ID="$1"
LOCAL_DIR="$2"
REVISION="${3:-main}"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: 'hf' CLI not found. Install huggingface_hub CLI first." >&2
  echo 'hint: python -m pip install -U "huggingface_hub[cli]"' >&2
  exit 1
fi

mkdir -p "${LOCAL_DIR}"

COMMON_ARGS=(
  download
  "${REPO_ID}"
  --local-dir "${LOCAL_DIR}"
  --revision "${REVISION}"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  COMMON_ARGS+=(--token "${HF_TOKEN}")
fi

# Ferrule current minimal local artifacts for llama-family models
FILES=(
  "config.json"
  "generation_config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "special_tokens_map.json"
  "model.safetensors"
)

echo "downloading ${REPO_ID}@${REVISION} -> ${LOCAL_DIR}"
hf "${COMMON_ARGS[@]}" "${FILES[@]}"

missing=0
for f in "${FILES[@]}"; do
  if [[ ! -f "${LOCAL_DIR}/${f}" ]]; then
    echo "missing required file: ${LOCAL_DIR}/${f}" >&2
    missing=1
  fi
done

if [[ ${missing} -ne 0 ]]; then
  echo "download incomplete" >&2
  exit 2
fi

echo "done."
echo "local_dir=${LOCAL_DIR}"