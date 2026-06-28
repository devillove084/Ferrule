#!/bin/bash
# Build Ferrule with CUDA support.
#   cargo oxide build --features cuda    (from workspace root)
set -e
cd "$(dirname "$0")/.."

echo "=== Building Ferrule with CUDA kernels ==="
cargo oxide build --features cuda

echo ""
echo "=== Done ==="
echo "Run: target/release/ferrule cuda           # probe GPU"
echo "Run: target/release/ferrule gpu-run ./models/OLMoE -p 'Hello' -n 4"
