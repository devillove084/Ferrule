#!/bin/sh
set -eu

# Keep this lane independent from the caller's CUDA/toolchain/target state.
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
CUDA_HOME=/home/admin/wxw403883/anaconda3/envs/tao
TOOLCHAIN_LINK="$ROOT/target/validation/qwen35-35b-current/toolchain/link"
TARGET_DIR="$ROOT/target/fp8-block32-cuda124"
MODE=${1:-run}

export CUDA_HOME
export NVCC="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$TOOLCHAIN_LINK:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:0:0
export FERRULE_CUDA_ARCH=sm_86
export FERRULE_CUTLASS_DIR="$ROOT/target/vendor/cutlass"
export CARGO_TARGET_DIR="$TARGET_DIR"
export RUSTFLAGS="-L native=$TOOLCHAIN_LINK -L native=$CUDA_HOME/lib"

if [ ! -x "$NVCC" ]; then
    echo "error: CUDA12.4 nvcc not found at $NVCC" >&2
    exit 1
fi
nvcc_version=$("$NVCC" --version)
printf '%s\n' "$nvcc_version"
case "$nvcc_version" in
    *"release 12.4"*|*"V12.4."*) ;;
    *)
        echo "error: expected CUDA12.4.x nvcc" >&2
        exit 1
        ;;
esac
if [ "$FERRULE_CUDA_ARCH" != sm_86 ]; then
    echo "error: FERRULE_CUDA_ARCH must be sm_86" >&2
    exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_info=$(nvidia-smi --id=0 \
        --query-gpu=name,compute_cap --format=csv,noheader,nounits)
    printf 'GPU[0]: %s\n' "$gpu_info"
    gpu_name=$(printf '%s\n' "$gpu_info" | cut -d, -f1 | sed 's/^ *//;s/ *$//')
    gpu_cap=$(printf '%s\n' "$gpu_info" | cut -d, -f2 | sed 's/^ *//;s/ *$//')
    case "$gpu_name" in
        *"RTX 3090"*) ;;
        *)
            echo "error: expected RTX 3090, got $gpu_name" >&2
            exit 1
            ;;
    esac
    if [ "$gpu_cap" != 8.6 ]; then
        echo "error: expected compute capability 8.6, got $gpu_cap" >&2
        exit 1
    fi
else
    echo "warning: nvidia-smi unavailable; GPU identity check skipped" >&2
fi

cd "$ROOT"
case "$MODE" in
    check)
        exec cargo check --locked --offline -p ferrule-backend --features cuda \
            --test fp8_block32_benchmark
        ;;
    run)
        exec cargo test --locked --offline -p ferrule-backend --features cuda \
            --test fp8_block32_benchmark -- \
            --exact fp8_block32_decode_bf16_gemm_baseline_vs_bf16_native \
            --ignored --nocapture --test-threads=1
        ;;
    *)
        echo "usage: $0 [check|run]" >&2
        exit 2
        ;;
esac
