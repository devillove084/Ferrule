#!/bin/sh
set -eu

CUTLASS_VERSION="v4.6.1"
CUTLASS_COMMIT="e05f953a5b3d38adc240df2ff928e0421c2abba3"
CUTLASS_REPOSITORY="https://github.com/NVIDIA/cutlass.git"
CUTLASS_DIR="${FERRULE_CUTLASS_DIR:-target/vendor/cutlass}"
VERSION_HEADER="$CUTLASS_DIR/include/cutlass/version.h"

verify_checkout() {
    if [ ! -d "$CUTLASS_DIR/.git" ]; then
        echo "error: $CUTLASS_DIR exists but is not a verifiable CUTLASS git checkout" >&2
        echo "remove it, or set FERRULE_CUTLASS_DIR to CUTLASS $CUTLASS_VERSION at $CUTLASS_COMMIT" >&2
        exit 1
    fi

    actual_commit=$(git --no-pager -C "$CUTLASS_DIR" rev-parse HEAD)
    if [ "$actual_commit" != "$CUTLASS_COMMIT" ]; then
        echo "error: CUTLASS checkout mismatch at $CUTLASS_DIR" >&2
        echo "expected: $CUTLASS_VERSION ($CUTLASS_COMMIT)" >&2
        echo "actual:   $actual_commit" >&2
        echo "remove the checkout and rerun this script" >&2
        exit 1
    fi

    dirty_paths=$(git --no-pager -C "$CUTLASS_DIR" status --porcelain --untracked-files=normal)
    if [ -n "$dirty_paths" ]; then
        echo "error: CUTLASS checkout at $CUTLASS_DIR contains local changes" >&2
        echo "$dirty_paths" >&2
        echo "restore the pinned checkout or use a clean FERRULE_CUTLASS_DIR" >&2
        exit 1
    fi

    if [ ! -f "$VERSION_HEADER" ]; then
        echo "error: verified CUTLASS checkout is missing $VERSION_HEADER" >&2
        exit 1
    fi

    if ! grep -q '^#define CUTLASS_MAJOR 4$' "$VERSION_HEADER" || \
       ! grep -q '^#define CUTLASS_MINOR 6$' "$VERSION_HEADER" || \
       ! grep -q '^#define CUTLASS_PATCH 1$' "$VERSION_HEADER"; then
        echo "error: CUTLASS version header does not declare 4.6.1" >&2
        exit 1
    fi
}

if [ -e "$CUTLASS_DIR" ]; then
    verify_checkout
    echo "→ CUTLASS $CUTLASS_VERSION ready at $CUTLASS_COMMIT"
    exit 0
fi

parent_dir=$(dirname "$CUTLASS_DIR")
temporary_dir="$CUTLASS_DIR.tmp.$$"
mkdir -p "$parent_dir"
rm -rf "$temporary_dir"
trap 'rm -rf "$temporary_dir"' EXIT HUP INT TERM

echo "→ fetching CUTLASS $CUTLASS_VERSION at $CUTLASS_COMMIT"
git init -q "$temporary_dir"
git -C "$temporary_dir" remote add origin "$CUTLASS_REPOSITORY"
git -C "$temporary_dir" fetch -q --depth 1 origin "$CUTLASS_COMMIT"
git -C "$temporary_dir" checkout -q --detach FETCH_HEAD
mv "$temporary_dir" "$CUTLASS_DIR"
trap - EXIT HUP INT TERM

verify_checkout
echo "→ CUTLASS $CUTLASS_VERSION ready at $CUTLASS_COMMIT"
