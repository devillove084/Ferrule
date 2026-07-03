# Ferrule build system
#   just build      → auto-detect CUDA + arch, build release
#   just build-cpu  → CPU-only
#   just build-cuda → CUDA build via cargo oxide with detected arch
#   just test-cuda  → CUDA tests via cargo oxide with detected arch
#   just run-cuda ARGS... → CUDA build via cargo oxide, then run target/release/ferrule
#   just chat MODEL → interactive chat (MODEL required)
#   just test       → all tests
#   just test-graph → compute graph IR tests

# ── Default ────────────────────────────────────────────────────────────

default: check test

# ── Detection helpers ──────────────────────────────────────────────────

# Detect CUDA compute capability from nvidia-smi. Override with FERRULE_CUDA_ARCH=sm_121.
# cargo-oxide must receive the exact arch; plain `cargo test -p ferrule-cuda` and
# plain `cargo run --features cuda` cannot link #[cuda_module] artifacts and may fail
# with cuda_oxide_artifact_anchor errors.
[private]
_cuda-arch := `if [ -n "${FERRULE_CUDA_ARCH:-}" ]; then echo "$FERRULE_CUDA_ARCH"; elif command -v nvidia-smi >/dev/null 2>&1; then cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '[:space:].'); if [ -n "${cap:-}" ]; then echo "sm_$cap"; else echo sm_86; fi; else echo sm_86; fi`

[private]
_has-oxide := `command -v cargo-oxide >/dev/null 2>&1 && echo 1 || echo 0`

[private]
_has-gpu := `command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0`

# Resolved: can we build/test CUDA through cargo-oxide?
[private]
_use-cuda := `if [ "${FERRULE_NO_CUDA:-}" = "1" ]; then echo 0; elif command -v cargo-oxide >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then echo 1; else echo 0; fi`

# ── Build ──────────────────────────────────────────────────────────────

build:
    @echo "=== Ferrule ==="
    @echo "oxide: $([ {{ _has-oxide }} = 1 ] && echo yes || echo no)"
    @echo "gpu:   $([ {{ _has-gpu }} = 1 ] && echo yes || echo no)"
    @echo "arch:  {{ _cuda-arch }}"
    @if [ "{{ _use-cuda }}" = "1" ]; then \
        echo "→ CUDA build via cargo oxide (arch: {{ _cuda-arch }})"; \
        cargo oxide build --features cuda --arch "{{ _cuda-arch }}" -- --release; \
    else \
        echo "→ CPU build"; \
        cargo build --release; \
    fi

build-cpu:
    cargo build --release

build-cuda arch='':
    @if [ "{{ _has-oxide }}" != "1" ]; then echo "error: cargo-oxide not found; run 'cargo install --path ...' or 'cargo oxide setup'"; exit 1; fi
    @arch="{{ arch }}"; test -n "$arch" || arch="{{ _cuda-arch }}"; echo "→ CUDA build via cargo oxide (arch: $arch)"; cargo oxide build --features cuda --arch "$arch" -- --release

build-dev:
    cargo build

# ── Check ──────────────────────────────────────────────────────────────

check:
    cargo check -p ferrule-cli

check-cuda:
    cargo check -p ferrule-cli --features cuda

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

test: test-graph test-runtime test-model test-bench test-cuda test-cli

test-graph:
    cargo test -p ferrule-graph

test-runtime:
    cargo test -p ferrule-runtime

test-model:
    cargo test -p ferrule-model

test-bench:
    cargo test -p ferrule-bench

test-cuda *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then \
        echo "→ CUDA tests skipped (oxide={{ _has-oxide }}, gpu={{ _has-gpu }}, FERRULE_NO_CUDA=$FERRULE_NO_CUDA)"; \
        echo "  Run 'just oxide-doctor' to configure CUDA tests, or 'just test-cuda-required' to fail when unavailable."; \
        exit 0; \
    fi
    @echo "→ CUDA tests via cargo oxide (arch: {{ _cuda-arch }})"
    cargo oxide test --arch {{ _cuda-arch }} -- -p ferrule-cuda {{ args }}

test-cuda-required *args='':
    @if [ "{{ _has-oxide }}" != "1" ]; then echo "error: cargo-oxide not found"; exit 1; fi
    @if [ "{{ _has-gpu }}" != "1" ]; then echo "error: no NVIDIA GPU detected"; exit 1; fi
    @echo "→ CUDA tests via cargo oxide (arch: {{ _cuda-arch }})"
    cargo oxide test --arch {{ _cuda-arch }} -- -p ferrule-cuda {{ args }}

test-cli:
    cargo test -p ferrule-cli

test-all: test
    @echo "=== All tests passed ==="

# ── Code quality ───────────────────────────────────────────────────────

fmt:
    cargo fmt -- --check

fmt-fix:
    cargo fmt

clippy:
    cargo clippy -p ferrule-core -p ferrule-graph -p ferrule-model -p ferrule-runtime -p ferrule-bench -p ferrule-cli -- -D warnings

clippy-cuda:
    cargo clippy -p ferrule-core -p ferrule-graph -p ferrule-model -p ferrule-runtime -p ferrule-cli --features cuda -- -D warnings

clippy-all: clippy clippy-cuda
    @echo "=== Clippy passed ==="

# ── Static analysis ────────────────────────────────────────────────────

audit:
    cargo audit

deny:
    -cargo deny check || true

udeps:
    cargo udeps

miri:
    cargo +nightly miri test -p ferrule-runtime

# ── MLIR (future kernel backend) ────────────────────────────────────────
# Replaces hand-written PTX with compiler-generated MLIR → NVVM/SPIR-V.
# Requires: LLVM 18+ with MLIR, https://mlir.llvm.org/
#
# mlir-check:
#     @which mlir-opt >/dev/null 2>&1 && echo "MLIR: $(mlir-opt --version)" || echo "MLIR: not installed"
# mlir-build:
#     cargo oxide build --features cuda,mlir

lint: fmt clippy-all
    @echo "=== Lint passed ==="

# ── Run ─────────────────────────────────────────────────────────────────

run-cuda *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ CUDA run via cargo oxide build (arch: {{ _cuda-arch }})"
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    ./target/release/ferrule {{ args }}

chat model quant='q4' *args='':
    just run-cuda chat {{ model }} -q {{ quant }} {{ args }}

run model prompt *args='':
    cargo run --release -p ferrule-cli -- run {{ model }} -p "{{ prompt }}" {{ args }}

gpu-run model prompt='Hello' n='16' quant='q4' *args='':
    just run-cuda gpu-run {{ model }} -p "{{ prompt }}" -n {{ n }} -q {{ quant }} {{ args }}

bench model prompt='Hello' n='128' quant='q4':
    just run-cuda bench-infer {{ model }} -p "{{ prompt }}" -n {{ n }} -q {{ quant }} --json

compare model prompt='The capital of France is' n='8' quant='q4':
    just run-cuda compare-logits {{ model }} -p "{{ prompt }}" -n {{ n }} -q {{ quant }}

# Real local DeepSeek-V4/DSpark HF shard probes. DSpark is not used in the base
# forward path; these run the DSV4-specific model boundary over local shards.
dsv4-probe prompt='Hello' layers='0' rows='4' topk='2' *args='':
    cargo run -p ferrule-cli -- deepseek-v4-probe models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --max-layers {{ layers }} --row-count {{ rows }} --top-k {{ topk }} {{ args }}

dsv4-topk prompt='Hello' layers='0' topk='3' chunk='4096' *args='':
    cargo run -p ferrule-cli -- deepseek-v4-probe models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --max-layers {{ layers }} --full-vocab-topk --top-k {{ topk }} --output-head-chunk-rows {{ chunk }} {{ args }}

dsv4-cuda-probe prompt='Hello' layers='1' rows='1' topk='0' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ CUDA run via cargo oxide build (arch: {{ _cuda-arch }})"
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    ./target/release/ferrule deepseek-v4-probe models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --backend cuda --max-layers {{ layers }} --row-count {{ rows }} --top-k {{ topk }} {{ args }}

dsv4-cuda-first-token prompt='Hello' topk='1' chunk='4096' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ CUDA run via cargo oxide build (arch: {{ _cuda-arch }})"
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    ./target/release/ferrule deepseek-v4-probe models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --backend cuda --max-layers 43 --full-vocab-topk --top-k {{ topk }} --output-head-chunk-rows {{ chunk }} {{ args }}

dsv4-cuda-generate prompt='Hello' tokens='4' chunk='4096' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ CUDA run via cargo oxide build (arch: {{ _cuda-arch }})"
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    ./target/release/ferrule deepseek-v4-generate models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --backend cuda --max-tokens {{ tokens }} --output-head-chunk-rows {{ chunk }} {{ args }}

dsv4-parity-json prompt='Hello' output='target/dsv4_generation_parity.json' *args='':
    python3 scripts/dsv4_generation_parity.py models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --output "{{ output }}" {{ args }}
    @echo "wrote {{ output }}"

dsv4-chat tokens='64' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @tokens="{{ tokens }}"; tokens="${tokens#tokens=}"; case "$tokens" in ''|*[!0-9]*) echo "error: dsv4-chat tokens must be an integer; use 'just dsv4-chat 64' or 'just dsv4-chat tokens=64'"; exit 2;; esac; echo "→ CUDA chat via cargo oxide build (arch: {{ _cuda-arch }}, tokens: $tokens)"; cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli; ./target/release/ferrule chat models/DeepSeek-V4-Flash-DSpark -q cuda -n "$tokens" --chat-template deepseek-v4 --temp 0 {{ args }}

serve model quant='q4' port='8080':
    just run-cuda server {{ model }} -q {{ quant }} --port {{ port }}

perplexity model file:
    cargo run --release -p ferrule-cli -- perplexity {{ model }} -f {{ file }}

info model:
    cargo run --release -p ferrule-cli -- info {{ model }}

inspect-cache path:
    cargo run --release -p ferrule-cli -- inspect-cache {{ path }}

# ── Clean ──────────────────────────────────────────────────────────────

clean:
    cargo clean
    rm -f ./*.o ./*.ptx ./*.ll ./*.opt.ll ./*.cubin ./*.fatbin ./*.sass
