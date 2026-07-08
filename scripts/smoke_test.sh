#!/bin/bash
set -e
BIN="${1:-./target/release/ferrule}"
MODEL="${2:-models/OLMoE-Instruct}"
echo "=== Ferrule Smoke Tests ==="
echo "Binary: $BIN"
echo "Model:  $MODEL"

# Test 1: info works
echo "--- Test 1: info ---"
$BIN info "$MODEL"

# Test 2: chat command exists and can start on CPU-compatible models.
# Some local checkpoints are CUDA-only or absent; callers can skip this with FERRULE_SKIP_CHAT_SMOKE=1.
if [ "${FERRULE_SKIP_CHAT_SMOKE:-0}" != "1" ]; then
  echo "--- Test 2: chat smoke ---"
  printf 'hi\n/exit\n' | timeout 120 $BIN chat "$MODEL" -q cpu -n 1 2>/dev/null >/tmp/ferrule-smoke-chat.out
  test -s /tmp/ferrule-smoke-chat.out && echo "PASS: chat produced output" || { echo "FAIL: chat produced no output"; exit 1; }
fi

# Test 3: inspect WeightPack if one exists next to the model.
CACHE=""
for candidate in "$MODEL"/*.weightpack "$MODEL"/*.qcache; do
  if [ -f "$candidate" ]; then
    CACHE="$candidate"
    break
  fi
done
if [ -n "$CACHE" ]; then
  echo "--- Test 3: inspect-weightpack ---"
  $BIN inspect-weightpack "$CACHE" && echo "PASS: inspect-weightpack" || { echo "FAIL: inspect-weightpack"; exit 1; }
fi

echo "=== Smoke tests passed ==="
