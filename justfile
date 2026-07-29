# Ferrule build system
#   just build      → auto-detect CUDA + arch, build release
#   just build-cuda → GB10 SM121a build via cargo oxide + CUTLASS
#   just test-cuda  → CUDA tests via cargo oxide with detected arch
#   just run-cuda ARGS... → CUDA build via cargo oxide, then run target/release/ferrule
#   just chat MODEL → interactive chat (MODEL required)
#   just bench-interactive MODEL → multi-turn chat latency benchmark
#   just dsv4-runtime-driver-bench → DSV4 benchmark through ResidentTopKDriver
#   just dsv4-runtime-driver-chunk-sweep → DSV4 prefill chunk sweep CSV/JSONL
#   just dsv4-serve → OpenAI-compatible DSV4 HTTP/SSE server
#   just dsv4-vllm-bench → vLLM official serving benchmark against a running server
#   just test       → CPU tests via nextest + doctests + optional CUDA tests

# ── Default ────────────────────────────────────────────────────────────

default: check test

# ── Detection helpers ──────────────────────────────────────────────────

# Detect CUDA compute capability from nvidia-smi. Override with FERRULE_CUDA_ARCH=sm_121.
# cargo-oxide must receive the exact arch; plain `cargo test -p ferrule-cuda` and
# plain `cargo run --features cuda` cannot link #[cuda_module] artifacts and may fail
# with cuda_oxide_artifact_anchor errors.
[private]
_cuda-arch := `if [ -n "${FERRULE_CUDA_ARCH:-}" ]; then echo "$FERRULE_CUDA_ARCH"; elif command -v nvidia-smi >/dev/null 2>&1; then cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '[:space:].'); if [ -n "${cap:-}" ]; then arch="sm_$cap"; case "$cap" in 10*|11*|12*) arch="${arch}a";; esac; echo "$arch"; else echo sm_86; fi; else echo sm_86; fi`

[private]
_has-oxide := `command -v cargo-oxide >/dev/null 2>&1 && echo 1 || echo 0`

[private]
_has-gpu := `command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0`

# Resolved: can we build/test CUDA through cargo-oxide?
[private]
_use-cuda := `if [ "${FERRULE_NO_CUDA:-}" = "1" ]; then echo 0; elif command -v cargo-oxide >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then echo 1; else echo 0; fi`

# ── Build ──────────────────────────────────────────────────────────────

build: cutlass-setup
    @echo "=== Ferrule GB10 ==="
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: Ferrule requires cargo-oxide and an NVIDIA GB10 GPU"; exit 1; fi
    @echo "arch: {{ _cuda-arch }}"
    cargo oxide build --features cuda --arch "{{ _cuda-arch }}" -- --release

build-cuda arch='': cutlass-setup
    @if [ "{{ _has-oxide }}" != "1" ]; then echo "error: cargo-oxide not found; run 'cargo install --path ...' or 'cargo oxide setup'"; exit 1; fi
    @arch="{{ arch }}"; test -n "$arch" || arch="{{ _cuda-arch }}"; echo "→ GB10 build via cargo oxide (arch: $arch)"; cargo oxide build --features cuda --arch "$arch" -- --release

# Idempotently fetch and verify the exact header-only CUTLASS revision. All GB10
# build/test/run recipes depend on this target; build.rs itself remains offline.
cutlass-setup:
    ./scripts/setup_cutlass.sh

# Explicit alias retained for discoverability; there is only one GB10 build path.
build-cutlass arch='':
    just build-cuda "{{ arch }}"

test-cutlass-provider arch='': cutlass-setup
    @if [ "{{ _has-oxide }}" != "1" ]; then echo "error: cargo-oxide not found"; exit 1; fi
    @if [ "{{ _has-gpu }}" != "1" ]; then echo "error: no NVIDIA GPU detected"; exit 1; fi
    @arch="{{ arch }}"; test -n "$arch" || arch="{{ _cuda-arch }}"; echo "→ CUTLASS provider tests via cargo oxide (arch: $arch)"; cargo oxide test --arch "$arch" -- -p ferrule-cuda --features cuda --test cutlass_provider

dsv4-dspark-attention-bench arch='': cutlass-setup
    @if [ "{{ _has-oxide }}" != "1" ]; then echo "error: cargo-oxide not found"; exit 1; fi
    @if [ "{{ _has-gpu }}" != "1" ]; then echo "error: no NVIDIA GPU detected"; exit 1; fi
    @arch="{{ arch }}"; test -n "$arch" || arch="{{ _cuda-arch }}"; echo "→ DSpark hybrid-attention latency via cargo oxide (arch: $arch)"; cargo oxide test --arch "$arch" -- -p ferrule-cuda --features cuda --test cutlass_provider sm121_dspark_hybrid_attention_formal_shape_latency -- --ignored --nocapture --test-threads=1

build-dev:
    cargo build

# ── Check ──────────────────────────────────────────────────────────────

check:
    cargo check --locked -p ferrule-cli

check-cuda:
    cargo check --locked -p ferrule-cli --features cuda

cuda-info:
    @echo "oxide: $([ {{ _has-oxide }} = 1 ] && echo yes || echo no)"
    @echo "gpu:   $([ {{ _has-gpu }} = 1 ] && echo yes || echo no)"
    @echo "arch:  {{ _cuda-arch }}"
    @echo "use:   $([ {{ _use-cuda }} = 1 ] && echo yes || echo no)"

oxide-doctor:
    @if [ "{{ _has-oxide }}" != "1" ]; then echo "error: cargo-oxide not found"; exit 1; fi
    cargo oxide doctor

oxide-build *args='':
    @if [ "{{ _has-oxide }}" != "1" ]; then echo "error: cargo-oxide not found"; exit 1; fi
    @echo "→ cargo oxide build --arch {{ _cuda-arch }} {{ args }}"
    cargo oxide build --arch {{ _cuda-arch }} {{ args }}

oxide-test *args='':
    @if [ "{{ _has-oxide }}" != "1" ]; then echo "error: cargo-oxide not found"; exit 1; fi
    @if [ "{{ _has-gpu }}" != "1" ]; then echo "error: no NVIDIA GPU detected; set FERRULE_NO_CUDA=1 to skip CUDA tests"; exit 1; fi
    @echo "→ cargo oxide test --arch {{ _cuda-arch }} -- {{ args }}"
    cargo oxide test --arch {{ _cuda-arch }} -- {{ args }}

# ── Test ───────────────────────────────────────────────────────────────

test: test-nextest test-docs test-cuda

test-nextest:
    @if ! command -v cargo-nextest >/dev/null 2>&1; then echo "error: cargo-nextest not found; run 'cargo install --locked cargo-nextest'"; exit 1; fi
    cargo nextest run --locked --workspace --exclude ferrule-cuda

test-docs:
    cargo test --locked --workspace --exclude ferrule-cuda --doc

test-runtime:
    cargo test --locked -p ferrule-runtime

test-model:
    cargo test --locked -p ferrule-model

test-server:
    cargo test --locked -p ferrule-server


test-cuda *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then \
        echo "→ CUDA tests skipped (oxide={{ _has-oxide }}, gpu={{ _has-gpu }}, FERRULE_NO_CUDA=$FERRULE_NO_CUDA)"; \
        echo "  Run 'just oxide-doctor' to configure CUDA tests, or 'just test-cuda-required' to fail when unavailable."; \
    else \
        echo "→ CUDA tests via cargo oxide (arch: {{ _cuda-arch }})"; \
        cargo oxide test --arch {{ _cuda-arch }} -- --features cuda -p ferrule-cuda {{ args }}; \
    fi

test-cuda-required *args='':
    @if [ "{{ _has-oxide }}" != "1" ]; then echo "error: cargo-oxide not found"; exit 1; fi
    @if [ "{{ _has-gpu }}" != "1" ]; then echo "error: no NVIDIA GPU detected"; exit 1; fi
    @echo "→ CUDA tests via cargo oxide (arch: {{ _cuda-arch }})"
    cargo oxide test --arch {{ _cuda-arch }} -- --features cuda -p ferrule-cuda {{ args }}

test-cli:
    cargo test --locked -p ferrule-cli

test-all: test
    @echo "=== All tests passed ==="

# ── Code quality ───────────────────────────────────────────────────────

fmt:
    cargo fmt --all -- --check

fmt-fix:
    cargo fmt

# ferrule-cuda's bindings require CUDA toolkit headers even without CUTLASS, so
# the platform-independent lane checks every CPU crate and leaves CUDA to the
# explicit CUDA lane below.
clippy:
    cargo clippy --locked --workspace --exclude ferrule-cuda --all-targets -- -D warnings

# Optional GB10/CUDA feature lint. This is intentionally separate from the
# platform-independent CI gate and uses the unified CUDA provider feature.
clippy-cuda:
    cargo clippy --locked -p ferrule-cli --all-targets --features cuda -- -D warnings

clippy-all: clippy clippy-cuda
    @echo "=== Clippy passed ==="

# ── Static analysis ────────────────────────────────────────────────────

audit:
    cargo audit

deny:
    cargo deny check

coverage:
    @if ! command -v cargo-nextest >/dev/null 2>&1; then echo "error: cargo-nextest not found; run 'cargo install --locked cargo-nextest'"; exit 1; fi
    @if ! command -v cargo-llvm-cov >/dev/null 2>&1; then echo "error: cargo-llvm-cov not found; run 'cargo install --locked cargo-llvm-cov'"; exit 1; fi
    rm -rf target/coverage
    mkdir -p target/coverage
    cargo llvm-cov nextest --locked --workspace --exclude ferrule-cuda --no-report
    cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
    cargo llvm-cov report --html --output-dir target/coverage
    cargo llvm-cov report --summary-only --output-path target/coverage/summary.txt --fail-under-lines 60

udeps:
    cargo udeps

miri:
    cargo miri test --locked --profile miri -p ferrule-runtime --lib

docs:
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --exclude ferrule-cuda --no-deps

# ── MLIR (future kernel backend) ────────────────────────────────────────
# Replaces hand-written PTX with compiler-generated MLIR → NVVM/SPIR-V.
# Requires: LLVM 18+ with MLIR, https://mlir.llvm.org/
#
# mlir-check:
#     @which mlir-opt >/dev/null 2>&1 && echo "MLIR: $(mlir-opt --version)" || echo "MLIR: not installed"
# mlir-build:
#     cargo oxide build --features cuda,mlir

lint: fmt clippy docs
    @echo "=== Lint passed ==="

# ── Run ─────────────────────────────────────────────────────────────────

run-cuda *args='': cutlass-setup
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: Ferrule requires cargo-oxide and an NVIDIA GB10 GPU"; exit 1; fi
    @echo "→ GB10 run via cargo oxide build (arch: {{ _cuda-arch }})"
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    ./target/release/ferrule {{ args }}

chat model quant='q4' *args='':
    just run-cuda chat {{ model }} -q {{ quant }} {{ args }}

bench-interactive model *args='':
    just run-cuda bench-interactive {{ model }} {{ args }}

dsv4-serve model='models/DeepSeek-V4-Flash-DSpark' port='8000' *args='':
    just run-cuda serve {{ model }} --host 127.0.0.1 --port {{ port }} --served-model-name deepseek-v4 {{ args }}

dsv4-vllm-bench mode='smoke' *args='':
    ./scripts/bench_vllm_serve.sh {{ mode }} {{ args }}


dsv4-runtime-driver-bench prompt1='Hello' prompt2='Explain Ferrule in one sentence.' tokens='1' warmup='1' chunk='4096' layers='43' *args='': cutlass-setup
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ DSV4 ResidentTopKDriver benchmark via CUDA + CUTLASS (arch: {{ _cuda-arch }}, tokens: {{ tokens }}, warmup: {{ warmup }}, chunk: {{ chunk }}, layers: {{ layers }})"
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    ./target/release/ferrule bench-interactive models/DeepSeek-V4-Flash-DSpark -p "{{ prompt1 }}" -p "{{ prompt2 }}" -n {{ tokens }} --warmup-tokens {{ warmup }} --prefill-chunk-size {{ chunk }} --max-layers {{ layers }} --json {{ args }}



# Sweep runtime-driver prefill chunk sizes and write CSV + JSONL under target/.
dsv4-runtime-driver-chunk-sweep chunks='1,2,4,8,16,4096' tokens='1' warmup='0' layers='43' output='target/bench/io-scheduler-e2e/chunks' sync='0' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ DSV4 ResidentTopKDriver chunk sweep (arch: {{ _cuda-arch }}, chunks: {{ chunks }}, tokens: {{ tokens }}, warmup: {{ warmup }}, layers: {{ layers }}, sync: {{ sync }})"
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    @sync_arg=""; if [ "{{ sync }}" = "1" ] || [ "{{ sync }}" = "true" ] || [ "{{ sync }}" = "sync" ]; then sync_arg="--profile-sync"; fi; python3 scripts/dsv4_runtime_driver_chunk_sweep.py --model models/DeepSeek-V4-Flash-DSpark --chunks "{{ chunks }}" --max-tokens {{ tokens }} --warmup-tokens {{ warmup }} --max-layers {{ layers }} --bin ./target/release/ferrule --output-dir {{ output }} $sync_arg {{ args }}



cuda:
    cargo run -p ferrule-cli -- cuda

inspect-weightpack path:
    cargo run -p ferrule-cli -- inspect-weightpack {{ path }}

expert-stream-smoke model layer='0' expert='0' *args='':
    cargo run -p ferrule-cli -- expert-stream-smoke {{ model }} --layer {{ layer }} --expert {{ expert }} {{ args }}



dsv4-cuda-generate prompt='Hello' tokens='4' chunk='4096' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ CUDA run via cargo oxide build (arch: {{ _cuda-arch }})"
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    ./target/release/ferrule deepseek-v4-generate models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --backend cuda --max-tokens {{ tokens }} --output-head-chunk-rows {{ chunk }} {{ args }}

dsv4-cuda-generate-json prompt='Hello' tokens='4' chunk='4096' output='target/dsv4-generate.json' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ GB10 JSON run via cargo oxide build (arch: {{ _cuda-arch }}, output: {{ output }})"
    @mkdir -p target
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    ./target/release/ferrule deepseek-v4-generate models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --backend cuda --max-tokens {{ tokens }} --output-head-chunk-rows {{ chunk }} --json {{ args }} | tee {{ output }}

dsv4-cuda-moe-profile prompt='Hello' tokens='4' chunk='4096' output='target/dsv4-moe-profile.json' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ GB10 MoE timing JSON run (arch: {{ _cuda-arch }}, output: {{ output }})"
    @mkdir -p target
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    FERRULE_CUDA_MOE_TIMING=1 ./target/release/ferrule deepseek-v4-generate models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --backend cuda --max-tokens {{ tokens }} --output-head-chunk-rows {{ chunk }} --json {{ args }} | tee {{ output }}



# Record platform support separately from throughput. A failed GDS check remains
# a failed recipe even though output is mirrored through tee.
dsv4-storage-platform-check output='target/bench/storage-platform-check.txt':
    @command -v gdscheck >/dev/null 2>&1 || { echo "error: gdscheck not found"; exit 1; }
    @mkdir -p "$(dirname "{{ output }}")"
    @bash -o pipefail -c '{ uname -a; echo; nvidia-smi; echo; gdscheck -p; } 2>&1 | tee "{{ output }}"'

dsv4-parity-json prompt='Hello' output='target/dsv4_generation_parity.json' *args='':
    python3 scripts/dsv4_generation_parity.py models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --output "{{ output }}" {{ args }}
    @echo "wrote {{ output }}"

dsv4-chat tokens='64' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @tokens="{{ tokens }}"; tokens="${tokens#tokens=}"; case "$tokens" in ''|*[!0-9]*) echo "error: dsv4-chat tokens must be an integer; use 'just dsv4-chat 64' or 'just dsv4-chat tokens=64'"; exit 2;; esac; echo "→ GB10 chat via cargo oxide build (arch: {{ _cuda-arch }}, tokens: $tokens)"; cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli; ./target/release/ferrule chat models/DeepSeek-V4-Flash-DSpark -q cuda -n "$tokens" --chat-template deepseek-v4 --temp 0 {{ args }}

info model:
    cargo run --release -p ferrule-cli -- info {{ model }}

# ── Clean ──────────────────────────────────────────────────────────────

clean:
    cargo clean
    rm -f ./*.o ./*.ptx ./*.ll ./*.opt.ll ./*.cubin ./*.fatbin ./*.sass
