#!/usr/bin/env bash
# Download models via ModelScope API — zero Python, uses curl + jq.
# Usage:
#   ./scripts/download.sh repo/name
#   ./scripts/download.sh repo/name --pat '*.safetensors'
#   ./scripts/download.sh repo/name --pat 'config.json,tokenizer.json'
set -euo pipefail

REPO=""; OUT="./models"; PAT=""; DRY=0; REV="master"
API="https://modelscope.cn/api/v1/models"

usage() { sed -n '2,6p' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --pat) PAT="$2"; shift 2 ;;
        --dry) DRY=1; shift ;;
        -h|--help) usage ;;
        -*) echo "Unknown: $1" >&2; exit 1 ;;
        *)  REPO="$1"; shift ;;
    esac
done

[[ -n "$REPO" ]] || { echo "Error: REPO required" >&2; usage; }
command -v curl &>/dev/null || { echo "Error: curl required" >&2; exit 1; }
command -v jq   &>/dev/null || { echo "Error: jq required" >&2; exit 1; }

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
TARGET="$OUT/$(basename "$REPO")"

# ── Fetch file list ──────────────────────────────────────────────────
echo "Fetching file list for $REPO..."
FILES_JSON=$(curl -sf "${API}/${REPO}/repo/files?Revision=${REV}&Recursive=true") || {
    echo "Error: failed to fetch file list" >&2; exit 1
}
echo "$FILES_JSON" | jq -e '.Data.Files' >/dev/null 2>&1 || {
    echo "Error: unexpected API response: $(echo "$FILES_JSON" | jq -r '.Message // "unknown"')" >&2; exit 1
}

TOTAL=$(echo "$FILES_JSON" | jq '.Data.Files | length')

# ── Filter files ─────────────────────────────────────────────────────
if [[ -n "$PAT" ]]; then
    # Comma-separated shell globs, e.g. '*.gguf,config.json'.
    MATCHING=$(echo "$FILES_JSON" | jq --arg pat "$PAT" '
      def trim: gsub("^\\s+|\\s+$"; "");
      def glob_re:
        gsub("\\."; "\\\\.")
        | gsub("\\*"; ".*")
        | gsub("\\?"; ".");
      ($pat | split(",") | map(trim) | map(select(length > 0)) | map("^" + (. | glob_re) + "$")) as $patterns
      | [.Data.Files[] | select(.Path as $path | any($patterns[]; $path | test(.)))]
    ')
else
    MATCHING=$(echo "$FILES_JSON" | jq '.Data.Files')
fi

COUNT=$(echo "$MATCHING" | jq 'length')
echo "Matched $COUNT / $TOTAL files"

if [[ "$DRY" -eq 1 ]]; then
    echo "$MATCHING" | jq -r '.[] | "  \(.Path)  \(.Size) bytes  LFS=\(.IsLFS)"'
    echo "Target: $TARGET"
    exit 0
fi

# ── Download files ────────────────────────────────────────────────────
mkdir -p "$TARGET"
DOWNLOADED=0; SKIPPED=0; FAILED=0

download_file() {
    local path="$1" size="$2" is_lfs="$3"
    local dest="$TARGET/$path"
    local dir; dir=$(dirname "$dest")
    mkdir -p "$dir"

    if [[ -f "$dest" ]]; then
        local existing; existing=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo 0)
        if [[ "$existing" -eq "$size" ]]; then
            ((SKIPPED++)) || true
            return 0
        fi
    fi

    local url="${API}/${REPO}/repo?Revision=${REV}&FilePath=${path}"
    if curl -sfL -o "$dest" "$url"; then
        ((DOWNLOADED++)) || true
        echo "  ok  $path"
    else
        ((FAILED++)) || true
        echo "  FAIL  $path" >&2
    fi
}

echo "Downloading to $TARGET..."
while IFS=$'\t' read -r path size is_lfs; do
    download_file "$path" "$size" "$is_lfs"
done < <(echo "$MATCHING" | jq -r '.[] | "\(.Path)\t\(.Size)\t\(.IsLFS)"')

echo "Done: $DOWNLOADED downloaded, $SKIPPED skipped, $FAILED failed → $TARGET"
[[ "$FAILED" -eq 0 ]] || exit 1
