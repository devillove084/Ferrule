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

# Test 2: CPU run produces expected text
echo "--- Test 2: CPU run ---"
OUT=$($BIN run "$MODEL" -p "The capital of France is" -n 4 2>/dev/null)
if echo "$OUT" | grep -qi "paris"; then
  echo "PASS: CPU run mentions Paris"
else
  echo "FAIL: CPU run didn't mention Paris. Got: $OUT"
  exit 1
fi

# Test 3: chat smoke (CPU)
echo "--- Test 3: CPU chat ---"
printf 'hi\n/exit\n' | timeout 120 $BIN chat "$MODEL" -q cpu -n 16 2>/dev/null | grep -qi "hello\|hi\|assist" && echo "PASS: CPU chat" || { echo "FAIL: CPU chat"; exit 1; }

# Test 4: inspect-cache (if cache exists)
CACHE=$(ls "$MODEL"/model.*.qcache 2>/dev/null | head -1)
if [ -n "$CACHE" ]; then
  echo "--- Test 4: inspect-cache ---"
  $BIN inspect-cache "$CACHE" && echo "PASS: inspect-cache" || { echo "FAIL: inspect-cache"; exit 1; }
fi

echo "=== All smoke tests passed ==="
