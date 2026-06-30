# Ferrule build system
#   just build      → auto-detect CUDA + arch, build release
#   just build-cpu  → CPU-only
#   just chat MODEL → interactive chat (MODEL required)
#   just test       → all tests

# ── Default ────────────────────────────────────────────────────────────

default: check test

# ── Detection helpers ──────────────────────────────────────────────────

# Detect CUDA compute capability from nvidia-smi, default to sm_86
_cuda-arch := `nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | sed 's/\.//' | xargs -I{} echo sm_{}`

_has-oxide := `command -v cargo-oxide >/dev/null 2>&1 && echo 1 || echo 0`

_has-gpu := `nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0`

# Resolved: can we build CUDA?
_use-cuda := `if [ "$$FERRULE_NO_CUDA" = "1" ]; then echo 0; elif [ "{{_has-oxide}}" = "1" ] && [ "{{_has-gpu}}" = "1" ]; then echo 1; else echo 0; fi`

# ── Build ──────────────────────────────────────────────────────────────

build:
    @echo "=== Ferrule ==="
    @echo "oxide: $([ {{_has-oxide}} = 1 ] && echo yes || echo no)"
    @echo "gpu:   $([ {{_has-gpu}} = 1 ] && echo yes || echo no)"
    @if [ "{{_use-cuda}}" = "1" ]; then \
        arch="{{_cuda-arch}}"; \
        test -n "$$arch" || arch="sm_86"; \
        echo "→ CUDA build (arch: $$arch)"; \
        cargo oxide build --release --features cuda --arch "$$arch"; \
    else \
        echo "→ CPU build"; \
        cargo build --release; \
    fi

build-cpu:
    cargo build --release

build-cuda arch=`{{_cuda-arch}}`:
    cargo oxide build --release --features cuda --arch {{arch}}

build-dev:
    cargo build

# ── Check ──────────────────────────────────────────────────────────────

check:
    cargo check -p ferrule-cli

check-cuda:
    cargo check -p ferrule-cli --features cuda

# ── Test ───────────────────────────────────────────────────────────────

test: test-runtime test-model test-cuda test-cli

test-runtime:
    cargo test -p ferrule-runtime

test-model:
    cargo test -p ferrule-model

test-cuda:
    cargo test -p ferrule-cuda

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
    cargo clippy -p ferrule-core -p ferrule-model -p ferrule-runtime -p ferrule-cli -- -D warnings

clippy-cuda:
    cargo clippy -p ferrule-core -p ferrule-model -p ferrule-runtime -p ferrule-cli --features cuda -- -D warnings

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

chat model quant='q4' *args='':
    cargo run --release -p ferrule-cli --features cuda -- chat {{model}} -q {{quant}} {{args}}

run model prompt *args='':
    cargo run --release -p ferrule-cli -- run {{model}} -p "{{prompt}}" {{args}}

gpu-run model prompt='Hello' n='16' quant='q4' *args='':
    cargo run --release -p ferrule-cli --features cuda -- gpu-run {{model}} -p "{{prompt}}" -n {{n}} -q {{quant}} {{args}}

bench model prompt='Hello' n='128' quant='q4':
    cargo run --release -p ferrule-cli --features cuda -- bench-infer {{model}} -p "{{prompt}}" -n {{n}} -q {{quant}} --json

compare model prompt='The capital of France is' n='8' quant='q4':
    cargo run --release -p ferrule-cli --features cuda -- compare-logits {{model}} -p "{{prompt}}" -n {{n}} -q {{quant}}

serve model quant='q4' port='8080':
    cargo run --release -p ferrule-cli --features cuda -- server {{model}} -q {{quant}} --port {{port}}

perplexity model file:
    cargo run --release -p ferrule-cli -- perplexity {{model}} -f {{file}}

info model:
    cargo run --release -p ferrule-cli -- info {{model}}

inspect-cache path:
    cargo run --release -p ferrule-cli -- inspect-cache {{path}}

# ── Clean ──────────────────────────────────────────────────────────────

clean:
    cargo clean
    rm -f ./*.o ./*.ptx ./*.ll ./*.opt.ll ./*.cubin ./*.fatbin ./*.sass
