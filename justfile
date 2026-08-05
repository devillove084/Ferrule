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

test: test-nextest test-docs test-cuda

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

test-cuda *args='':
    @if [ "{{ _use-cuda }}" = "1" ]; then arch="{{ _cuda-arch }}"; echo "→ CUDA backend tests (arch: $arch)"; FERRULE_CUDA_ARCH="$arch" cargo test --locked -p ferrule-backend --features cuda {{ args }} -- --test-threads=1; else echo "→ CUDA tests skipped (nvcc={{ _has-nvcc }}, gpu={{ _has-gpu }}, FERRULE_NO_CUDA=${FERRULE_NO_CUDA:-})"; echo "  Run 'just test-cuda-required' to require CUDA."; fi

test-cuda-required *args='': cutlass-setup
    @test "{{ _has-nvcc }}" = "1" || { echo "error: nvcc not found"; exit 1; }; test "{{ _has-gpu }}" = "1" || { echo "error: no NVIDIA GPU detected"; exit 1; }; arch="{{ _cuda-arch }}"; test -n "$arch" || { echo "error: could not detect CUDA architecture; set FERRULE_CUDA_ARCH"; exit 1; }; echo "→ CUDA backend tests (arch: $arch)"; FERRULE_CUDA_ARCH="$arch" cargo test --locked -p ferrule-backend --features cuda {{ args }} -- --test-threads=1

test-cli:
    cargo test --locked -p ferrule-cli

test-all: test
    @echo "=== All tests passed ==="

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
    cargo llvm-cov nextest --locked --workspace --no-report
    cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
    cargo llvm-cov report --html --output-dir target/coverage
    cargo llvm-cov report --summary-only --output-path target/coverage/summary.txt --fail-under-lines 60

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

dsv4-runtime-driver-chunk-sweep chunks='1,2,4,8,16,4096' tokens='1' warmup='0' layers='43' output='target/bench/io-scheduler-e2e/chunks' sync='0' *args='': cutlass-setup
    @test "{{ _use-cuda }}" = "1" || { echo "error: CUDA run requires nvcc and an NVIDIA GPU"; exit 1; }; arch="{{ _cuda-arch }}"; FERRULE_CUDA_ARCH="$arch" cargo build --locked --release -p ferrule-cli --features cuda
    @sync_arg=""; if [ "{{ sync }}" = "1" ] || [ "{{ sync }}" = "true" ] || [ "{{ sync }}" = "sync" ]; then sync_arg="--profile-sync"; fi; python3 scripts/dsv4_runtime_driver_chunk_sweep.py --model models/DeepSeek-V4-Flash-0731 --chunks "{{ chunks }}" --max-tokens {{ tokens }} --warmup-tokens {{ warmup }} --max-layers {{ layers }} --bin ./target/release/ferrule --output-dir {{ output }} $sync_arg {{ args }}

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

dsv4-parity-json prompt='Hello' output='target/dsv4_generation_parity.json' *args='':
    python3 scripts/dsv4_generation_parity.py models/DeepSeek-V4-Flash-0731 --prompt "{{ prompt }}" --output "{{ output }}" {{ args }}
    @echo "wrote {{ output }}"

dsv4-chat tokens='64' *args='':
    @tokens="{{ tokens }}"; tokens="${tokens#tokens=}"; case "$tokens" in ''|*[!0-9]*) echo "error: dsv4-chat tokens must be an integer"; exit 2;; esac; just run-cuda chat models/DeepSeek-V4-Flash-0731 -q cuda -n "$tokens" --chat-template deepseek-v4 --temp 0 {{ args }}

info model:
    cargo run --release -p ferrule-cli -- info {{ model }}

# ── Clean ──────────────────────────────────────────────────────────────

clean:
    cargo clean
    rm -f ./*.o ./*.ptx ./*.ll ./*.opt.ll ./*.cubin ./*.fatbin ./*.sass
