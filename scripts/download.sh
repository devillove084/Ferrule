#!/usr/bin/env bash
# Download models via ModelScope API.
# Dependencies: curl, jq.
# Optional dependency: aria2c for faster parallel downloads.
#
# Usage:
#   ./scripts/download.sh repo/name
#   ./scripts/download.sh repo/name --pat '*.safetensors'
#   ./scripts/download.sh repo/name --pat 'config.json,tokenizer.json'
#   ./scripts/download.sh repo/name --out ./models
#   ./scripts/download.sh repo/name --rev master
#   ./scripts/download.sh repo/name --dry
#
# Tuning:
#   JOBS=4 CONN=8 RETRY=10 ./scripts/download.sh repo/name

set -euo pipefail

REPO=""
OUT="./models"
PAT=""
DRY=0
REV="master"
API="https://modelscope.cn/api/v1/models"

usage() {
    sed -n '2,15p' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            OUT="$2"
            shift 2
            ;;
        --pat)
            PAT="$2"
            shift 2
            ;;
        --rev)
            REV="$2"
            shift 2
            ;;
        --dry)
            DRY=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Error: unknown option: $1" >&2
            exit 1
            ;;
        *)
            REPO="$1"
            shift
            ;;
    esac
done

[[ -n "$REPO" ]] || {
    echo "Error: REPO required" >&2
    usage
}

command -v curl >/dev/null 2>&1 || {
    echo "Error: curl required" >&2
    exit 1
}

command -v jq >/dev/null 2>&1 || {
    echo "Error: jq required" >&2
    exit 1
}

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
TARGET="$OUT/$(basename "$REPO")"

JOBS="${JOBS:-4}"
CONN="${CONN:-8}"
RETRY="${RETRY:-10}"

urlencode() {
    jq -rn --arg v "$1" '$v | @uri'
}

is_complete() {
    local path="$1"
    local size="$2"
    local dest="$TARGET/$path"

    [[ -f "$dest" ]] || return 1

    local existing
    existing=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo 0)

    [[ "$existing" -eq "$size" ]]
}

build_download_url() {
    local path="$1"
    local encoded_path
    local encoded_rev

    encoded_path="$(urlencode "$path")"
    encoded_rev="$(urlencode "$REV")"

    printf '%s/%s/repo?Revision=%s&FilePath=%s' "$API" "$REPO" "$encoded_rev" "$encoded_path"
}

echo "Fetching file list for $REPO..."

ENCODED_REV="$(urlencode "$REV")"

FILES_JSON=$(curl -sfL "${API}/${REPO}/repo/files?Revision=${ENCODED_REV}&Recursive=true") || {
    echo "Error: failed to fetch file list" >&2
    exit 1
}

FILES=$(echo "$FILES_JSON" | jq -c '
  def payload:
    if has("Data") then .Data
    elif has("data") then .data
    else null
    end;

  payload as $d
  | if ($d | type) == "object" and ($d.Files | type) == "array" then
      $d.Files
    elif ($d | type) == "object" and ($d.files | type) == "array" then
      $d.files
    elif ($d | type) == "array" then
      $d
    else
      empty
    end
') || {
    echo "Error: failed to parse API response" >&2
    echo "$FILES_JSON" | jq '.' >&2
    exit 1
}

[[ -n "$FILES" && "$FILES" != "null" ]] || {
    echo "Error: unexpected API response:" >&2
    echo "$FILES_JSON" | jq '.' >&2
    exit 1
}

FILES=$(echo "$FILES" | jq -c '
  [
    .[]
    | {
        Path: (.Path // .path // .Name // .name),
        Size: ((.Size // .size // .FileSize // .file_size // 0) | tonumber? // 0),
        IsLFS: (.IsLFS // .is_lfs // .LFS // .lfs // false),
        Type: (.Type // .type // "")
      }
    | select(.Path != null)
    | select((.Path | tostring | endswith("/")) | not)
    | select((.Type | tostring | ascii_downcase) != "tree")
    | select((.Type | tostring | ascii_downcase) != "directory")
    | select((.Type | tostring | ascii_downcase) != "dir")
    | select((.Type | tostring | ascii_downcase) != "folder")
  ]
')

TOTAL=$(echo "$FILES" | jq 'length')

if [[ -n "$PAT" ]]; then
    MATCHING=$(echo "$FILES" | jq --arg pat "$PAT" '
      def trim:
        gsub("^\\s+|\\s+$"; "");

      def glob_re:
        gsub("\\."; "\\\\.")
        | gsub("\\+"; "\\\\+")
        | gsub("\\("; "\\\\(")
        | gsub("\\)"; "\\\\)")
        | gsub("\\["; "\\\\[")
        | gsub("\\]"; "\\\\]")
        | gsub("\\{"; "\\\\{")
        | gsub("\\}"; "\\\\}")
        | gsub("\\^"; "\\\\^")
        | gsub("\\$"; "\\\\$")
        | gsub("\\|"; "\\\\|")
        | gsub("\\*"; ".*")
        | gsub("\\?"; ".");

      ($pat
        | split(",")
        | map(trim)
        | map(select(length > 0))
        | map("^" + (. | glob_re) + "$")
      ) as $patterns
      | [.[] | select(.Path as $path | any($patterns[]; $path | test(.)))]
    ')
else
    MATCHING=$(echo "$FILES" | jq '.')
fi

COUNT=$(echo "$MATCHING" | jq 'length')

echo "Matched $COUNT / $TOTAL files"

if [[ "$COUNT" -eq 0 ]]; then
    echo "Error: no files matched" >&2
    exit 1
fi

if [[ "$DRY" -eq 1 ]]; then
    echo "$MATCHING" | jq -r '.[] | "  \(.Path)  \(.Size) bytes  LFS=\(.IsLFS)"'
    echo "Target: $TARGET"
    exit 0
fi

mkdir -p "$TARGET"

echo "Downloading to $TARGET..."
echo "Parallel files: $JOBS, connections per file: $CONN"

if command -v aria2c >/dev/null 2>&1; then
    URI_FILE="$(mktemp)"
    trap 'rm -f "$URI_FILE"' EXIT

    SKIPPED=0
    TODO=0

    while IFS=$'\t' read -r path size is_lfs; do
        if is_complete "$path" "$size"; then
            ((SKIPPED++)) || true
            continue
        fi

        mkdir -p "$(dirname "$TARGET/$path")"

        url="$(build_download_url "$path")"

        {
            printf '%s\n' "$url"
            printf '  out=%s\n' "$path"
        } >> "$URI_FILE"

        ((TODO++)) || true
    done < <(echo "$MATCHING" | jq -r '.[] | "\(.Path)\t\(.Size)\t\(.IsLFS)"')

    if [[ "$TODO" -eq 0 ]]; then
        echo "Done: all $SKIPPED files already complete -> $TARGET"
        exit 0
    fi

    aria2c \
        --input-file="$URI_FILE" \
        --dir="$TARGET" \
        --continue=true \
        --allow-overwrite=true \
        --auto-file-renaming=false \
        --max-concurrent-downloads="$JOBS" \
        --split="$CONN" \
        --max-connection-per-server="$CONN" \
        --min-split-size=16M \
        --file-allocation=none \
        --max-tries="$RETRY" \
        --retry-wait=3 \
        --connect-timeout=10 \
        --timeout=60 \
        --lowest-speed-limit=50K

    FAILED=0
    COMPLETE=0

    while IFS=$'\t' read -r path size is_lfs; do
        if is_complete "$path" "$size"; then
            ((COMPLETE++)) || true
        else
            echo "  FAIL or incomplete: $path" >&2
            ((FAILED++)) || true
        fi
    done < <(echo "$MATCHING" | jq -r '.[] | "\(.Path)\t\(.Size)\t\(.IsLFS)"')

    echo "Done: $COMPLETE complete, $SKIPPED pre-skipped, $FAILED failed -> $TARGET"
    [[ "$FAILED" -eq 0 ]] || exit 1

else
    echo "aria2c not found, falling back to curl serial mode." >&2
    echo "Install aria2c for faster downloads." >&2

    DOWNLOADED=0
    SKIPPED=0
    FAILED=0

    download_file() {
        local path="$1"
        local size="$2"
        local is_lfs="$3"
        local dest="$TARGET/$path"
        local dir
        local url

        dir="$(dirname "$dest")"
        mkdir -p "$dir"

        if is_complete "$path" "$size"; then
            ((SKIPPED++)) || true
            return 0
        fi

        url="$(build_download_url "$path")"

        if curl -fL \
            -C - \
            --retry "$RETRY" \
            --retry-all-errors \
            --retry-delay 3 \
            --connect-timeout 10 \
            --speed-limit 50K \
            --speed-time 60 \
            -o "$dest" \
            "$url"; then

            if is_complete "$path" "$size"; then
                ((DOWNLOADED++)) || true
                echo "  ok  $path"
            else
                ((FAILED++)) || true
                echo "  FAIL size mismatch  $path" >&2
            fi
        else
            ((FAILED++)) || true
            echo "  FAIL  $path" >&2
        fi
    }

    while IFS=$'\t' read -r path size is_lfs; do
        download_file "$path" "$size" "$is_lfs"
    done < <(echo "$MATCHING" | jq -r '.[] | "\(.Path)\t\(.Size)\t\(.IsLFS)"')

    echo "Done: $DOWNLOADED downloaded, $SKIPPED skipped, $FAILED failed -> $TARGET"
    [[ "$FAILED" -eq 0 ]] || exit 1
fi
