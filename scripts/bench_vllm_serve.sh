#!/usr/bin/env bash
# Benchmark a running Ferrule OpenAI server with vLLM's official serving harness.
#
# Usage:
#   ./scripts/bench_vllm_serve.sh smoke
#   ./scripts/bench_vllm_serve.sh baseline
#   ./scripts/bench_vllm_serve.sh sweep
#
# Common overrides:
#   BASE_URL=http://127.0.0.1:8000 \
#   TOKENIZER=models/DeepSeek-V4-Flash-DSpark \
#   INPUT_LEN=32 OUTPUT_LEN=8 NUM_PROMPTS=20 \
#   CONCURRENCIES=1,2,4 \
#   ./scripts/bench_vllm_serve.sh sweep
#
# Any arguments after MODE are forwarded to `vllm bench serve`.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
    sed -n '2,17p' "$0"
}

MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then
    shift
fi
EXTRA_ARGS=("$@")

case "$MODE" in
    smoke)
        DEFAULT_INPUT_LEN=16
        DEFAULT_OUTPUT_LEN=2
        DEFAULT_NUM_PROMPTS=2
        DEFAULT_CONCURRENCIES=1
        ;;
    baseline)
        DEFAULT_INPUT_LEN=32
        DEFAULT_OUTPUT_LEN=8
        DEFAULT_NUM_PROMPTS=20
        DEFAULT_CONCURRENCIES=1
        ;;
    sweep)
        DEFAULT_INPUT_LEN=32
        DEFAULT_OUTPUT_LEN=8
        DEFAULT_NUM_PROMPTS=20
        DEFAULT_CONCURRENCIES=1,2,4
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "error: unknown mode '$MODE' (expected smoke, baseline, or sweep)" >&2
        usage >&2
        exit 2
        ;;
esac

command -v vllm >/dev/null 2>&1 || {
    echo "error: vllm CLI not found in PATH" >&2
    exit 1
}
command -v curl >/dev/null 2>&1 || {
    echo "error: curl is required" >&2
    exit 1
}

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"
MODEL="${MODEL:-deepseek-v4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
TOKENIZER="${TOKENIZER:-models/DeepSeek-V4-Flash-DSpark}"
INPUT_LEN="${INPUT_LEN:-$DEFAULT_INPUT_LEN}"
OUTPUT_LEN="${OUTPUT_LEN:-$DEFAULT_OUTPUT_LEN}"
NUM_PROMPTS="${NUM_PROMPTS:-$DEFAULT_NUM_PROMPTS}"
CONCURRENCIES="${CONCURRENCIES:-$DEFAULT_CONCURRENCIES}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
SEED="${SEED:-0}"
READY_TIMEOUT_SECS="${READY_TIMEOUT_SECS:-600}"
METRIC_PERCENTILES="${METRIC_PERCENTILES:-50,90,95,99}"
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_ROOT="${RESULT_ROOT:-target/bench/vllm-serve/$TIMESTAMP}"

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: $name must be a positive integer, got '$value'" >&2
        exit 2
    fi
}

require_positive_integer INPUT_LEN "$INPUT_LEN"
require_positive_integer OUTPUT_LEN "$OUTPUT_LEN"
require_positive_integer NUM_PROMPTS "$NUM_PROMPTS"
require_positive_integer READY_TIMEOUT_SECS "$READY_TIMEOUT_SECS"
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "error: SEED must be a non-negative integer, got '$SEED'" >&2
    exit 2
fi

if [[ ! -e "$TOKENIZER" ]]; then
    echo "warning: tokenizer '$TOKENIZER' is not a local path; vLLM may download it" >&2
fi

mkdir -p "$RESULT_ROOT"
RESULT_ROOT="$(cd "$RESULT_ROOT" && pwd)"

VLLM_VERSION="$(vllm --version 2>&1 | tr '\n' ' ')"
GIT_COMMIT="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"

curl -fsS "$BASE_URL/health" >"$RESULT_ROOT/health.txt" || {
    echo "error: Ferrule health check failed at $BASE_URL/health" >&2
    echo "start the server first, for example:" >&2
    echo "  FERRULE_DSV4_PROFILE=false FERRULE_DSV4_PROFILE_SYNC=false FERRULE_CUDA_MOE_TIMING=false just dsv4-serve" >&2
    exit 1
}
curl -fsS "$BASE_URL/v1/models" >"$RESULT_ROOT/models.json"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >"$RESULT_ROOT/nvidia-smi.txt" 2>&1 || true
    nvidia-smi \
        --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory \
        --format=csv,noheader \
        >"$RESULT_ROOT/gpu-summary.csv" 2>&1 || true
fi

cat >"$RESULT_ROOT/run.env" <<EOF
mode=$MODE
utc_timestamp=$TIMESTAMP
base_url=$BASE_URL
endpoint=$ENDPOINT
model=$MODEL
served_model_name=$SERVED_MODEL_NAME
tokenizer=$TOKENIZER
input_len=$INPUT_LEN
output_len=$OUTPUT_LEN
num_prompts=$NUM_PROMPTS
concurrencies=$CONCURRENCIES
request_rate=$REQUEST_RATE
seed=$SEED
metric_percentiles=$METRIC_PERCENTILES
bench_cuda_visible_devices=$BENCH_CUDA_VISIBLE_DEVICES
vllm_version=$VLLM_VERSION
ferrule_commit=$GIT_COMMIT
ferrule_dirty_files=$GIT_DIRTY
EOF

printf '%s\n' "Ferrule vLLM serving benchmark" \
    "  mode:          $MODE" \
    "  server:        $BASE_URL$ENDPOINT" \
    "  model:         $SERVED_MODEL_NAME" \
    "  tokenizer:     $TOKENIZER" \
    "  input/output:  $INPUT_LEN/$OUTPUT_LEN tokens" \
    "  prompts:       $NUM_PROMPTS per run" \
    "  concurrency:   $CONCURRENCIES" \
    "  request rate:  $REQUEST_RATE" \
    "  client CUDA:   ${BENCH_CUDA_VISIBLE_DEVICES:-hidden}" \
    "  results:       $RESULT_ROOT"

IFS=',' read -r -a CONCURRENCY_VALUES <<<"$CONCURRENCIES"
for concurrency in "${CONCURRENCY_VALUES[@]}"; do
    concurrency="${concurrency//[[:space:]]/}"
    if [[ -z "$concurrency" || ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: invalid concurrency '$concurrency' in CONCURRENCIES=$CONCURRENCIES" >&2
        exit 2
    fi

    minimum_prompts=$((concurrency * 5))
    if [[ "$MODE" != "smoke" ]] && (( NUM_PROMPTS < minimum_prompts )); then
        echo "warning: NUM_PROMPTS=$NUM_PROMPTS is below 5x concurrency ($minimum_prompts); percentiles will be noisy" >&2
    fi

    label="${MODE}-c${concurrency}-in${INPUT_LEN}-out${OUTPUT_LEN}"
    result_filename="$label.json"
    log_filename="$RESULT_ROOT/$label.log"

    command=(
        vllm bench serve
        --backend openai-chat
        --base-url "$BASE_URL"
        --endpoint "$ENDPOINT"
        --model "$MODEL"
        --served-model-name "$SERVED_MODEL_NAME"
        --tokenizer "$TOKENIZER"
        --dataset-name random
        --random-input-len "$INPUT_LEN"
        --random-output-len "$OUTPUT_LEN"
        --random-range-ratio 0
        --num-prompts "$NUM_PROMPTS"
        --request-rate "$REQUEST_RATE"
        --max-concurrency "$concurrency"
        --temperature 0
        --ignore-eos
        --seed "$SEED"
        --ready-check-timeout-sec "$READY_TIMEOUT_SECS"
        --percentile-metrics ttft,tpot,itl,e2el
        --metric-percentiles "$METRIC_PERCENTILES"
        --save-result
        --save-detailed
        --result-dir "$RESULT_ROOT"
        --result-filename "$result_filename"
        --metadata
        "ferrule_commit=$GIT_COMMIT"
        "benchmark_mode=$MODE"
        "max_concurrency=$concurrency"
        "input_len=$INPUT_LEN"
        "output_len=$OUTPUT_LEN"
    )
    command+=("${EXTRA_ARGS[@]}")

    echo
    echo "=== Running concurrency=$concurrency ==="
    printf 'command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    CUDA_VISIBLE_DEVICES="$BENCH_CUDA_VISIBLE_DEVICES" "${command[@]}" 2>&1 | tee "$log_filename"
done

ln -sfn "$RESULT_ROOT" "$(dirname "$RESULT_ROOT")/latest"
echo
echo "Benchmark complete: $RESULT_ROOT"
echo "Attach run.env, gpu-summary.csv, *.json, and *.log when sharing results."
