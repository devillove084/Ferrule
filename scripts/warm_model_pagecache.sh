#!/usr/bin/env bash
# Warm Linux page cache by reading model files from local or network storage.
#
# Usage:
#   ./scripts/warm_model_pagecache.sh DeepSeek-V4-Flash-0731
#   ./scripts/warm_model_pagecache.sh /mnt/nas1/hf/DeepSeek-V4-Flash-0731
#   ./scripts/warm_model_pagecache.sh --jobs 16 DeepSeek-V4-Flash-0731
#   ./scripts/warm_model_pagecache.sh --dry-run DeepSeek-V4-Flash-0731
#
# Defaults and overrides:
#   MODEL_ROOT=/mnt/nas1/hf
#   JOBS=logical CPU count reported by nproc
#   BLOCK_SIZE=16M

set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/mnt/nas1/hf}"
JOBS="${JOBS:-}"
BLOCK_SIZE="${BLOCK_SIZE:-16M}"
MODEL=""
DRY_RUN=0
QUIET=0

usage() {
    sed -n '2,13p' "$0"
}

die() {
    echo "error: $*" >&2
    exit 2
}

require_value() {
    local option="$1"
    local remaining="$2"
    (( remaining >= 2 )) || die "$option requires a value"
}

human_bytes() {
    local bytes="$1"
    if command -v numfmt >/dev/null 2>&1; then
        numfmt --to=iec-i --suffix=B "$bytes"
    else
        printf '%s bytes\n' "$bytes"
    fi
}

while (( $# > 0 )); do
    case "$1" in
        -j|--jobs)
            require_value "$1" "$#"
            JOBS="$2"
            shift 2
            ;;
        --root)
            require_value "$1" "$#"
            MODEL_ROOT="$2"
            shift 2
            ;;
        --block-size)
            require_value "$1" "$#"
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -q|--quiet)
            QUIET=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            die "unknown option '$1'"
            ;;
        *)
            [[ -z "$MODEL" ]] || die "only one model path may be specified"
            MODEL="$1"
            shift
            ;;
    esac
done

if (( $# > 0 )); then
    [[ -z "$MODEL" ]] || die "only one model path may be specified"
    (( $# == 1 )) || die "only one model path may be specified"
    MODEL="$1"
fi

[[ -n "$MODEL" ]] || {
    usage >&2
    die "model name or path is required"
}
[[ -n "$BLOCK_SIZE" ]] || die "BLOCK_SIZE must not be empty"

for command in find xargs stat dd readlink awk date nproc; do
    command -v "$command" >/dev/null 2>&1 || die "$command is required"
done

[[ -n "$JOBS" ]] || JOBS="$(nproc)"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "JOBS must be a positive integer, got '$JOBS'"

if [[ "$MODEL" = /* ]]; then
    candidate="$MODEL"
else
    candidate="$MODEL_ROOT/$MODEL"
fi

MODEL_PATH="$(readlink -f -- "$candidate" 2>/dev/null || true)"
[[ -n "$MODEL_PATH" && ( -d "$MODEL_PATH" || -f "$MODEL_PATH" ) ]] || \
    die "model path does not exist: $candidate"

MANIFEST="$(mktemp "${TMPDIR:-/tmp}/ferrule-pagecache.XXXXXX")"
cleanup() {
    rm -f "$MANIFEST"
}
trap cleanup EXIT HUP INT TERM

if [[ -f "$MODEL_PATH" ]]; then
    printf '%s\0' "$MODEL_PATH" >"$MANIFEST"
elif ! find -L "$MODEL_PATH" \
    -type d -name .git -prune -o \
    -type f -print0 >"$MANIFEST"; then
    die "failed to enumerate model files under $MODEL_PATH"
fi

read -r FILE_COUNT TOTAL_BYTES < <(
    xargs -0 -r stat --dereference --format='%s' <"$MANIFEST" |
        awk '{ count += 1; bytes += $1 } END { printf "%d %.0f\n", count, bytes }'
)

(( FILE_COUNT > 0 )) || die "no regular files found under $MODEL_PATH"

AVAILABLE_BYTES=0
if [[ -r /proc/meminfo ]]; then
    AVAILABLE_BYTES="$(
        awk '/^MemAvailable:/ { printf "%.0f\n", $2 * 1024; exit }' /proc/meminfo
    )"
fi

printf '%s\n' \
    "Ferrule model page-cache warmup" \
    "  model:       $MODEL_PATH" \
    "  files:       $FILE_COUNT" \
    "  bytes:       $(human_bytes "$TOTAL_BYTES")" \
    "  workers:     $JOBS" \
    "  block size:  $BLOCK_SIZE"

if (( AVAILABLE_BYTES > 0 )); then
    echo "  memory free: $(human_bytes "$AVAILABLE_BYTES")"
    if (( TOTAL_BYTES > AVAILABLE_BYTES )); then
        echo "warning: model size exceeds currently available memory; the kernel may evict earlier pages before warmup completes" >&2
    fi
fi

if (( DRY_RUN == 1 )); then
    echo "Dry run complete; no model payload was read."
    exit 0
fi

export BLOCK_SIZE QUIET
START_TIME="$(date +%s)"

if ! xargs -0 -r -n 1 -P "$JOBS" bash -c '
    path="$1"
    dd if="$path" of=/dev/null bs="$BLOCK_SIZE" status=none
    if [[ "$QUIET" != "1" ]]; then
        printf "  cached: %s\n" "$path"
    fi
' _ <"$MANIFEST"; then
    echo "error: one or more model files could not be read" >&2
    exit 1
fi

END_TIME="$(date +%s)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
(( ELAPSED_SECONDS > 0 )) || ELAPSED_SECONDS=1
BYTES_PER_SECOND=$((TOTAL_BYTES / ELAPSED_SECONDS))

printf '%s\n' \
    "Warmup complete" \
    "  elapsed:     ${ELAPSED_SECONDS}s" \
    "  throughput:  $(human_bytes "$BYTES_PER_SECOND")/s"
