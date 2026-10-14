#!/bin/sh
set -eu
if [ "${1-}" != '--sole-gpu-owner-granted' ] || [ "$#" != 3 ]; then
  echo 'Usage from repository root: run-after-grant.sh --sole-gpu-owner-granted PREPARED_DIR FRESH_RECEIPTS_DIR' >&2
  exit 2
fi
# The parent stays GPU-hidden; only each owned job receives its frozen environment.
# Old jobs must be reprepared, never edited in place to bypass changed source pins.
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void
exec /opt/conda310/bin/python -B scripts/run_pinned_jobs.py \
  --sole-gpu-owner-granted --jobs "$2/jobs.json" --output "$3" \
  --libraries target/validation/bf16-layer01-supported-retry-20261005/environment-preflight.json
