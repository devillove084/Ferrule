# Ferrule build system
#   just build      → auto-detect CUDA and build release
#   just build-cuda → standard Cargo + NVCC CUDA build
#   just test-cuda  → CUDA backend tests with the detected architecture
#   just run-cuda ARGS... → CUDA release build, then run ferrule
#   just test       → workspace tests, doctests, and optional CUDA tests

# ── Default ────────────────────────────────────────────────────────────

default: check test

# ── Detection helpers ──────────────────────────────────────────────────

# `FERRULE_CUDA_ARCH` always wins for compilation. Otherwise convert the first
# GPU's compute capability literally: 10.3 becomes sm_103. Architecture-specific
# suffixes (`a`/`f`) are never inferred from the major version.
[private]
_cuda-arch := `if [ -n "${FERRULE_CUDA_ARCH:-}" ]; then printf '%s\n' "$FERRULE_CUDA_ARCH"; elif command -v nvidia-smi >/dev/null 2>&1; then cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '[:space:].'); if [ -n "$cap" ]; then printf 'sm_%s\n' "$(printf '%s' "$cap" | tr -d '.')"; fi; fi`

# Runtime validation must ignore `FERRULE_CUDA_ARCH`: a binary compiled for a
# different target must never be launched merely because some NVIDIA GPU exists.
[private]
_cuda-device-arch := `if command -v nvidia-smi >/dev/null 2>&1; then cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '[:space:].'); if [ -n "$cap" ]; then printf 'sm_%s\n' "$(printf '%s' "$cap" | tr -d '.')"; fi; fi`

[private]
_has-nvcc := `command -v nvcc >/dev/null 2>&1 && echo 1 || echo 0`

[private]
_has-gpu := `command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0`

[private]
_use-cuda := `if [ "${FERRULE_NO_CUDA:-}" = "1" ]; then echo 0; elif command -v nvcc >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then echo 1; else echo 0; fi`

# ── Build ──────────────────────────────────────────────────────────────

build:
    @if [ "{{ _use-cuda }}" = "1" ]; then just build-cuda; else echo "→ CPU release build"; cargo build --locked --release; fi

build-cuda arch='': cutlass-setup
    @arch="{{ arch }}"; test -n "$arch" || arch="{{ _cuda-arch }}"; test "{{ _has-nvcc }}" = "1" || { echo "error: nvcc not found"; exit 1; }; test "{{ _has-gpu }}" = "1" || { echo "error: no NVIDIA GPU detected"; exit 1; }; test -n "$arch" || { echo "error: could not detect CUDA architecture; set FERRULE_CUDA_ARCH"; exit 1; }; echo "→ CUDA release build (arch: $arch)"; FERRULE_CUDA_ARCH="$arch" cargo build --locked --release --features cuda

cutlass-setup:
    ./scripts/setup_cutlass.sh

build-cutlass arch='':
    just build-cuda "{{ arch }}"

# Compile every CUDA backend target without loading it on the local GPU. Each
# architecture gets an isolated Cargo target directory so build-script outputs
# and native objects cannot be reused across incompatible capabilities.
check-cuda-arch arch: cutlass-setup
    @test "{{ _has-nvcc }}" = "1" || { echo "error: nvcc not found"; exit 1; }; test -n "{{ arch }}" || { echo "error: CUDA architecture is required"; exit 1; }; echo "→ CUDA compile-only validation (arch: {{ arch }})"; CUDA_VISIBLE_DEVICES="" CARGO_TARGET_DIR="target/validation/{{ arch }}" FERRULE_CUDA_ARCH="{{ arch }}" cargo test --locked --release -p ferrule-backend --features cuda --all-targets --no-run

check-cuda-arch-matrix arches='sm_89 sm_90 sm_103 sm_120': cutlass-setup
    @for arch in {{ arches }}; do just check-cuda-arch "$arch" || exit 1; done

test-cutlass-provider arch='': cutlass-setup
    @arch="{{ arch }}"; test -n "$arch" || arch="{{ _cuda-arch }}"; device_arch="{{ _cuda-device-arch }}"; test "{{ _has-nvcc }}" = "1" || { echo "error: nvcc not found"; exit 1; }; test "{{ _has-gpu }}" = "1" || { echo "error: no NVIDIA GPU detected"; exit 1; }; test -n "$arch" || { echo "error: could not determine requested CUDA architecture"; exit 1; }; test "$arch" = "$device_arch" || { echo "error: refusing to run $arch provider tests on $device_arch hardware; use 'just check-cuda-arch $arch' for compile-only validation"; exit 1; }; FERRULE_CUDA_ARCH="$arch" cargo test --locked -p ferrule-backend --features cuda --test cutlass_provider -- --test-threads=1

proposal-hybrid-attention-bench arch='': cutlass-setup
    @arch="{{ arch }}"; test -n "$arch" || arch="{{ _cuda-arch }}"; device_arch="{{ _cuda-device-arch }}"; test "{{ _has-nvcc }}" = "1" || { echo "error: nvcc not found"; exit 1; }; test "{{ _has-gpu }}" = "1" || { echo "error: no NVIDIA GPU detected"; exit 1; }; test "$arch" = "$device_arch" || { echo "error: refusing to run $arch benchmark on $device_arch hardware"; exit 1; }; FERRULE_CUDA_ARCH="$arch" cargo test --locked -p ferrule-backend --features cuda --test cutlass_provider hybrid_attention_formal_shape_latency -- --ignored --nocapture --test-threads=1

build-dev:
    cargo build --locked

# ── Check ──────────────────────────────────────────────────────────────

check:
    cargo check --locked --workspace --all-targets

check-cuda: cutlass-setup
    @test "{{ _has-nvcc }}" = "1" || { echo "error: nvcc not found"; exit 1; }; arch="{{ _cuda-arch }}"; test -n "$arch" || { echo "error: could not detect CUDA architecture; set FERRULE_CUDA_ARCH"; exit 1; }; FERRULE_CUDA_ARCH="$arch" cargo check --locked -p ferrule-cli --features cuda --all-targets

cuda-info:
    @echo "nvcc:  $([ {{ _has-nvcc }} = 1 ] && echo yes || echo no)"
    @echo "gpu:   $([ {{ _has-gpu }} = 1 ] && echo yes || echo no)"
    @echo "arch:  {{ _cuda-arch }}"
    @echo "use:   $([ {{ _use-cuda }} = 1 ] && echo yes || echo no)"
    @if command -v nvcc >/dev/null 2>&1; then nvcc --version; fi

# ── Test ───────────────────────────────────────────────────────────────

# Local convenience only: optional CUDA may remain UNVERIFIED. CI uses ci-cpu.
test: test-nextest test-docs test-cuda

# Required CPU lane: every non-ignored workspace target plus doctests. Do not
# exclude model tests to hide missing fixtures; hermetic failures must stay red.
ci-cpu:
    @command -v cargo-nextest >/dev/null 2>&1 || { echo "error: required cargo-nextest missing (JUnit cannot be produced)"; exit 1; }
    FERRULE_NO_CUDA=1 CUDA_VISIBLE_DEVICES="" CARGO_TARGET_DIR=target cargo nextest run --locked --workspace --all-targets --profile ci --no-tests fail
    FERRULE_NO_CUDA=1 CUDA_VISIBLE_DEVICES="" CARGO_TARGET_DIR=target cargo test --locked --workspace --doc

# Compile AND link all workspace CUDA targets, including both native providers.
# CUDA stubs may be supplied through LIBRARY_PATH on a GPU-less devel image;
# never put stubs in LD_LIBRARY_PATH or run these binaries as GPU evidence.
[positional-arguments]
ci-cuda-compile arch='sm_86':
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p target/validation/ci/cuda-compile
    echo 'FAILED_OR_INTERRUPTED: CUDA compilation not verified' > target/validation/ci/cuda-compile/status.txt
    case "${FERRULE_NO_CUDA:-0}" in 0) ;; *) echo 'error: CUDA compile lane requires FERRULE_NO_CUDA unset or 0'; exit 1;; esac
    case "$1" in sm_[0-9][0-9]|sm_[0-9][0-9][0-9]) ;; *) echo 'error: expected explicit sm_XX or sm_XXX'; exit 1;; esac
    command -v nvcc >/dev/null || { echo 'error: nvcc missing'; exit 1; }
    just cutlass-setup
    export FERRULE_CUDA_ARCH="$1" FERRULE_NO_CUDA=0 CUDA_VISIBLE_DEVICES=""
    export CARGO_TARGET_DIR="target/validation/cuda-compile/$1"
    cargo check --locked --workspace --features cuda --all-targets
    cargo test --locked --workspace --features cuda --all-targets --no-run
    echo 'VERIFIED_COMPILE_ONLY: feature/native/CUTLASS all-targets check and link; GPU execution UNVERIFIED' > target/validation/ci/cuda-compile/status.txt

# A successful probe is only a prerequisite, never evidence of GPU tests passing.
# Numeric visible-device ordinals are intentional: no implicit selection of a
# different GPU when CUDA_VISIBLE_DEVICES is empty, hidden, or malformed.
[positional-arguments]
ci-gpu-preflight arch='sm_86' devices='1':
    #!/usr/bin/env bash
    set -euo pipefail
    case "${FERRULE_NO_CUDA:-0}" in 0) ;; *) echo 'UNVERIFIED: GPU lane requires FERRULE_NO_CUDA unset or 0'; exit 1;; esac
    command -v nvcc >/dev/null || { echo 'UNVERIFIED: nvcc missing'; exit 1; }
    command -v nvidia-smi >/dev/null || { echo 'UNVERIFIED: nvidia-smi missing'; exit 1; }
    case "$2" in 1|4) ;; *) echo 'error: supported device requirements are 1 or 4'; exit 1;; esac
    query=(--query-gpu=compute_cap --format=csv,noheader)
    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
        [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'UNVERIFIED: CUDA_VISIBLE_DEVICES must expose explicit numeric ordinals'; exit 1; }
        IFS=, read -ra ordinals <<< "$CUDA_VISIBLE_DEVICES"
        declare -A seen=()
        for ordinal in "${ordinals[@]}"; do
            [[ "$ordinal" = 0 || "$ordinal" =~ ^[1-9][0-9]*$ ]] || { echo 'UNVERIFIED: noncanonical CUDA ordinal'; exit 1; }
            [[ ! -v seen[$ordinal] ]] || { echo 'UNVERIFIED: duplicate CUDA ordinal'; exit 1; }
            seen[$ordinal]=1
        done
        query+=(--id="$CUDA_VISIBLE_DEVICES")
    fi
    caps=$(nvidia-smi "${query[@]}") || { echo 'UNVERIFIED: GPU enumeration failed'; exit 1; }
    count=0
    while IFS= read -r cap; do
        cap=${cap//[[:space:]]/}
        [[ "sm_${cap//./}" = "$1" ]] || { echo "UNVERIFIED: requested $1 does not match visible capability $cap"; exit 1; }
        count=$((count + 1))
    done <<< "$caps"
    (( count >= $2 )) || { echo "UNVERIFIED: requires $2 GPUs, found $count"; exit 1; }
    echo 'GPU prerequisites present; CUDA initialization and execution must still pass the selected tests'

[positional-arguments]
ci-gpu-smoke arch='sm_86':
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p target/validation/ci/gpu-smoke
    echo 'FAILED_OR_INTERRUPTED: required GPU smoke not verified' > target/validation/ci/gpu-smoke/status.txt
    rm -f target/nextest/gpu-smoke/junit.xml
    just ci-gpu-preflight "$1" 1
    command -v cargo-nextest >/dev/null || { echo 'error: required cargo-nextest missing'; exit 1; }
    just cutlass-setup
    FERRULE_NO_CUDA=0 FERRULE_CUDA_ARCH="$1" CARGO_TARGET_DIR=target cargo nextest run --locked -p ferrule-backend --features cuda --profile gpu-smoke --run-ignored all --no-tests fail
    echo 'VERIFIED: selected required GPU smoke tests passed' > target/validation/ci/gpu-smoke/status.txt

# Optional nightly classification is performed BEFORE starting tests. No model
# download and no blanket ignored run. Once started, every test failure is fatal.
ci-gpu-nightly-preflight arch='sm_86': (ci-gpu-preflight arch '4')
    @test -n "${FERRULE_QWEN35_08B_DIR:-}" && test -r "$FERRULE_QWEN35_08B_DIR/config.json" || { echo 'UNVERIFIED: explicit local Qwen3.5-0.8B fixture required'; exit 1; }
    @test -n "${FERRULE_QWEN35_ORACLE_DIR:-}" && test -r "$FERRULE_QWEN35_ORACLE_DIR/manifest.json" || { echo 'UNVERIFIED: explicit Qwen3.5 oracle manifest required'; exit 1; }

[positional-arguments]
ci-gpu-nightly arch='sm_86':
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p target/validation/ci/gpu-nightly
    echo 'FAILED_OR_INTERRUPTED: nightly GPU execution not verified' > target/validation/ci/gpu-nightly/status.txt
    rm -f target/nextest/gpu-nightly/junit.xml
    just ci-gpu-nightly-preflight "$1"
    command -v cargo-nextest >/dev/null || { echo 'error: cargo-nextest missing'; exit 1; }
    just cutlass-setup
    FERRULE_NO_CUDA=0 FERRULE_CUDA_ARCH="$1" CARGO_TARGET_DIR=target cargo nextest run --locked --workspace --features cuda --profile gpu-nightly --run-ignored only --no-tests fail
    echo 'VERIFIED: selected nightly GPU tests passed (not all ignored tests)' > target/validation/ci/gpu-nightly/status.txt

test-nextest:
    @if command -v cargo-nextest >/dev/null 2>&1; then cargo nextest run --locked --workspace; else cargo test --locked --workspace --all-targets; fi

test-docs:
    cargo test --locked --workspace --doc

test-runtime:
    cargo test --locked -p ferrule-runtime

test-model:
    cargo test --locked -p ferrule-model

test-server:
    cargo test --locked -p ferrule-server

# These entry points now select the audited smoke profile, including ignored
# contracts; arbitrary Cargo filters must not bypass the required selection.
test-cuda:
    @case "${FERRULE_NO_CUDA:-0}" in 0|1) ;; *) echo 'error: FERRULE_NO_CUDA must be 0 or 1'; exit 1;; esac
    @if [ "{{ _use-cuda }}" = "1" ]; then just ci-gpu-smoke "{{ _cuda-arch }}"; else echo "→ UNVERIFIED: optional CUDA tests not run (nvcc={{ _has-nvcc }}, gpu={{ _has-gpu }}, FERRULE_NO_CUDA=${FERRULE_NO_CUDA:-})"; echo "  Run 'just test-cuda-required' to require CUDA."; fi

test-cuda-required:
    just ci-gpu-smoke "{{ _cuda-arch }}"

test-cli:
    cargo test --locked -p ferrule-cli

test-all: test
    @echo "=== Selected tests completed; optional CUDA may be UNVERIFIED (see above) ==="

# ── Code quality ───────────────────────────────────────────────────────

fmt:
    cargo fmt --all -- --check

fmt-fix:
    cargo fmt --all

clippy:
    cargo clippy --locked --workspace --all-targets -- -D warnings

clippy-cuda: cutlass-setup
    @test "{{ _has-nvcc }}" = "1" || { echo "error: nvcc not found"; exit 1; }; arch="{{ _cuda-arch }}"; test -n "$arch" || { echo "error: could not detect CUDA architecture; set FERRULE_CUDA_ARCH"; exit 1; }; FERRULE_CUDA_ARCH="$arch" cargo clippy --locked -p ferrule-cli --all-targets --features cuda -- -D warnings

clippy-all: clippy clippy-cuda
    @echo "=== Clippy passed ==="

# ── Static analysis ────────────────────────────────────────────────────

audit:
    cargo audit

deny:
    cargo deny check

coverage:
    @if ! command -v cargo-nextest >/dev/null 2>&1; then echo "error: cargo-nextest not found"; exit 1; fi
    @if ! command -v cargo-llvm-cov >/dev/null 2>&1; then echo "error: cargo-llvm-cov not found"; exit 1; fi
    rm -rf target/coverage
    mkdir -p target/coverage
    FERRULE_NO_CUDA=1 CUDA_VISIBLE_DEVICES="" CARGO_TARGET_DIR=target cargo llvm-cov nextest --locked --workspace --all-targets --profile ci --no-report
    CARGO_TARGET_DIR=target cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
    CARGO_TARGET_DIR=target cargo llvm-cov report --html --output-dir target/coverage
    CARGO_TARGET_DIR=target cargo llvm-cov report --summary-only --output-path target/coverage/summary.txt --fail-under-lines 60

udeps:
    cargo udeps

miri:
    cargo miri test --locked --profile miri -p ferrule-runtime --lib

docs:
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps

lint: fmt clippy docs
    @echo "=== Lint passed ==="

# ── Run ────────────────────────────────────────────────────────────────

run-cuda *args='': cutlass-setup
    @test "{{ _use-cuda }}" = "1" || { echo "error: CUDA run requires nvcc and an NVIDIA GPU"; exit 1; }; arch="{{ _cuda-arch }}"; echo "→ CUDA release build (arch: $arch)"; FERRULE_CUDA_ARCH="$arch" cargo build --locked --release -p ferrule-cli --features cuda
    ./target/release/ferrule {{ args }}

chat model quant='q4' *args='':
    just run-cuda chat {{ model }} -q {{ quant }} {{ args }}

bench-interactive model *args='':
    just run-cuda bench-interactive {{ model }} {{ args }}

dsv4-serve model='models/DeepSeek-V4-Flash-0731' port='8000' *args='':
    just run-cuda serve {{ model }} --host 127.0.0.1 --port {{ port }} --served-model-name deepseek-v4 {{ args }}

dsv4-vllm-bench mode='smoke' *args='':
    ./scripts/bench_vllm_serve.sh {{ mode }} {{ args }}

dsv4-runtime-driver-bench prompt1='Hello' prompt2='Explain Ferrule in one sentence.' tokens='1' warmup='1' chunk='4096' layers='43' *args='':
    just run-cuda bench-interactive models/DeepSeek-V4-Flash-0731 -p "{{ prompt1 }}" -p "{{ prompt2 }}" -n {{ tokens }} --warmup-tokens {{ warmup }} --prefill-chunk-size {{ chunk }} --max-layers {{ layers }} --json {{ args }}

cuda:
    cargo run -p ferrule-cli -- cuda

inspect-weightpack path:
    cargo run -p ferrule-cli -- inspect-weightpack {{ path }}

expert-stream-smoke model layer='0' expert='0' *args='':
    cargo run -p ferrule-cli -- expert-stream-smoke {{ model }} --layer {{ layer }} --expert {{ expert }} {{ args }}

dsv4-cuda-generate prompt='Hello' tokens='4' chunk='4096' *args='':
    just run-cuda deepseek-v4-generate models/DeepSeek-V4-Flash-0731 --prompt "{{ prompt }}" --max-tokens {{ tokens }} --output-head-chunk-rows {{ chunk }} {{ args }}

dsv4-cuda-generate-json prompt='Hello' tokens='4' chunk='4096' output='target/dsv4-generate.json' *args='':
    @mkdir -p target
    just run-cuda deepseek-v4-generate models/DeepSeek-V4-Flash-0731 --prompt "{{ prompt }}" --max-tokens {{ tokens }} --output-head-chunk-rows {{ chunk }} --json {{ args }} | tee {{ output }}

dsv4-cuda-moe-profile prompt='Hello' tokens='4' chunk='4096' output='target/dsv4-moe-profile.json' *args='': cutlass-setup
    @test "{{ _use-cuda }}" = "1" || { echo "error: CUDA run requires nvcc and an NVIDIA GPU"; exit 1; }; mkdir -p target; arch="{{ _cuda-arch }}"; FERRULE_CUDA_ARCH="$arch" cargo build --locked --release -p ferrule-cli --features cuda; FERRULE_CUDA_MOE_TIMING=1 ./target/release/ferrule deepseek-v4-generate models/DeepSeek-V4-Flash-0731 --prompt "{{ prompt }}" --max-tokens {{ tokens }} --output-head-chunk-rows {{ chunk }} --json {{ args }} | tee {{ output }}

dsv4-storage-platform-check output='target/bench/storage-platform-check.txt':
    @command -v gdscheck >/dev/null 2>&1 || { echo "error: gdscheck not found"; exit 1; }
    @mkdir -p "$(dirname "{{ output }}")"
    @bash -o pipefail -c '{ uname -a; echo; nvidia-smi; echo; gdscheck -p; } 2>&1 | tee "{{ output }}"'

dsv4-chat tokens='64' *args='':
    @tokens="{{ tokens }}"; tokens="${tokens#tokens=}"; case "$tokens" in ''|*[!0-9]*) echo "error: dsv4-chat tokens must be an integer"; exit 2;; esac; just run-cuda chat models/DeepSeek-V4-Flash-0731 -q cuda -n "$tokens" --chat-template deepseek-v4 --temp 0 {{ args }}

info model:
    cargo run --release -p ferrule-cli -- info {{ model }}

# ── Clean ──────────────────────────────────────────────────────────────

clean:
    cargo clean
    rm -f ./*.o ./*.ptx ./*.ll ./*.opt.ll ./*.cubin ./*.fatbin ./*.sass
