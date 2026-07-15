# Ferrule build system
#   just build      → auto-detect CUDA + arch, build release
#   just build-cpu  → CPU-only
#   just build-cuda → CUDA build via cargo oxide with detected arch
#   just test-cuda  → CUDA tests via cargo oxide with detected arch
#   just run-cuda ARGS... → CUDA build via cargo oxide, then run target/release/ferrule
#   just chat MODEL → interactive chat (MODEL required)
#   just bench-interactive MODEL → multi-turn chat latency benchmark
#   just dsv4-runtime-driver-bench → DSV4 benchmark through ResidentTopKDriver
#   just dsv4-resident-roofline → DSV4 exact resident/no-I/O target-pass roofline
#   just dsv4-serve → OpenAI-compatible DSV4 HTTP/SSE server
#   just dsv4-vllm-bench → vLLM official serving benchmark against a running server
#   just dsv4-prefill-chunk-sweep → DSV4 runtime-driver prefill chunk sweep CSV/JSONL
#   just test       → all workspace tests
#   just test-graph → runtime graph IR tests

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

# Fetch the pinned header-only CUTLASS checkout into ignored build artifacts.
cutlass-setup:
    @if [ -f target/vendor/cutlass/include/cutlass/version.h ]; then echo "→ CUTLASS 4.6.1 headers already present"; else mkdir -p target/vendor; git clone --depth 1 --branch v4.6.1 https://github.com/NVIDIA/cutlass.git target/vendor/cutlass; fi

# Build cuda-oxide control kernels plus CUTLASS leaf providers for this GPU.
build-cutlass arch='': cutlass-setup
    @if [ "{{ _has-oxide }}" != "1" ]; then echo "error: cargo-oxide not found"; exit 1; fi
    @arch="{{ arch }}"; test -n "$arch" || arch="{{ _cuda-arch }}"; echo "→ CUDA + CUTLASS build (arch: $arch)"; cargo oxide build --features cuda,cutlass --arch "$arch" -- --release

test-cutlass-provider: cutlass-setup
    cargo test -p ferrule-cuda --features cutlass --test cutlass_provider

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

test: test-graph test-runtime test-model test-server test-cuda test-cli

test-graph:
    cargo test -p ferrule-runtime graph

test-runtime:
    cargo test -p ferrule-runtime

test-model:
    cargo test -p ferrule-model

test-server:
    cargo test -p ferrule-server


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
    cargo clippy -p ferrule-common -p ferrule-model -p ferrule-runtime -p ferrule-server -p ferrule-cli -- -D warnings

clippy-cuda:
    cargo clippy -p ferrule-common -p ferrule-model -p ferrule-runtime -p ferrule-server -p ferrule-cli --features cuda -- -D warnings

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

bench-interactive model *args='':
    just run-cuda bench-interactive {{ model }} {{ args }}

dsv4-serve model='models/DeepSeek-V4-Flash-DSpark' port='8000' *args='':
    just run-cuda serve {{ model }} --host 127.0.0.1 --port {{ port }} --served-model-name deepseek-v4 {{ args }}

dsv4-vllm-bench mode='smoke' *args='':
    ./scripts/bench_vllm_serve.sh {{ mode }} {{ args }}


dsv4-runtime-driver-bench prompt1='Hello' prompt2='Explain Ferrule in one sentence.' tokens='1' warmup='1' chunk='4096' layers='43' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ DSV4 ResidentTopKDriver benchmark via cargo oxide (arch: {{ _cuda-arch }}, tokens: {{ tokens }}, warmup: {{ warmup }}, chunk: {{ chunk }}, layers: {{ layers }})"
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    ./target/release/ferrule bench-interactive models/DeepSeek-V4-Flash-DSpark -p "{{ prompt1 }}" -p "{{ prompt2 }}" -n {{ tokens }} --warmup-tokens {{ warmup }} --prefill-chunk-size {{ chunk }} --max-layers {{ layers }} --json {{ args }}

# Exact target-pass roofline: capture selected experts once, then replay the same
# decode with independent KV states and require zero selected expert I/O.
dsv4-resident-roofline model='models/DeepSeek-V4-Flash-DSpark' prompt='Hello' layers='43' hotset='48' chunk='4096' output='target/bench/s0-profile/resident-replay.json' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ DSV4 resident/no-I/O roofline (arch: {{ _cuda-arch }}, layers: {{ layers }}, hotset: {{ hotset }})"
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    mkdir -p target/bench/s0-profile
    FERRULE_EXPERT_IO_BACKEND=io_uring FERRULE_EXPERT_IO_QUEUE_DEPTH=2 FERRULE_EXPERT_IO_BUFFER_MIB=16 FERRULE_EXPERT_IO_SLABS=16 ./target/release/ferrule bench-interactive {{ model }} -p "{{ prompt }}" --max-layers {{ layers }} --output-head-chunk-rows {{ chunk }} --moe-prefetch-experts 0 --moe-hotset-experts {{ hotset }} --resident-replay --json {{ args }} | tee {{ output }}

# Gate F1: reuse bench-interactive's resident harness for native
# 1-sequence × V-row verification at V=2/4/8, including the output head.
dsv4-verify-width-sweep model='models/DeepSeek-V4-Flash-DSpark' prompt='Explain Ferrule runtime architecture in one concise paragraph.' layers='43' hotset='48' chunk='4096' iterations='3' output='target/bench/gate-f1/verify-width-sweep.json' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ DSV4 Gate F1 resident verification width sweep (arch: {{ _cuda-arch }}, V=2/4/8, iterations: {{ iterations }})"
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    mkdir -p target/bench/gate-f1
    FERRULE_EXPERT_IO_BACKEND=io_uring FERRULE_EXPERT_IO_QUEUE_DEPTH=2 FERRULE_EXPERT_IO_BUFFER_MIB=16 FERRULE_EXPERT_IO_SLABS=16 ./target/release/ferrule bench-interactive {{ model }} -p "{{ prompt }}" --max-layers {{ layers }} --output-head-chunk-rows {{ chunk }} --moe-prefetch-experts 0 --moe-hotset-experts {{ hotset }} --verify-width-sweep --verify-iterations {{ iterations }} --json {{ args }} | tee {{ output }}

# Sweep runtime-driver prefill chunk sizes and write CSV + JSONL under target/.
dsv4-runtime-driver-chunk-sweep chunks='1,2,4,8,16,4096' tokens='1' warmup='0' layers='43' output='target/dsv4-runtime-driver-chunk-sweep' sync='0' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ DSV4 ResidentTopKDriver chunk sweep (arch: {{ _cuda-arch }}, chunks: {{ chunks }}, tokens: {{ tokens }}, warmup: {{ warmup }}, layers: {{ layers }}, sync: {{ sync }})"
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    @sync_arg=""; if [ "{{ sync }}" = "1" ] || [ "{{ sync }}" = "true" ] || [ "{{ sync }}" = "sync" ]; then sync_arg="--profile-sync"; fi; python3 scripts/dsv4_runtime_driver_chunk_sweep.py --model models/DeepSeek-V4-Flash-DSpark --chunks "{{ chunks }}" --max-tokens {{ tokens }} --warmup-tokens {{ warmup }} --max-layers {{ layers }} --bin ./target/release/ferrule --output-dir {{ output }} $sync_arg {{ args }}

# Short alias focused on Step-C prefill observation.
dsv4-prefill-chunk-sweep chunks='1,2,4,8,16,4096' tokens='1' warmup='0' layers='43' output='target/dsv4-runtime-driver-chunk-sweep' sync='0' *args='':
    just dsv4-runtime-driver-chunk-sweep "{{ chunks }}" "{{ tokens }}" "{{ warmup }}" "{{ layers }}" "{{ output }}" "{{ sync }}" {{ args }}

test-dsv4-runtime-driver-local *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: local DSV4 runtime-driver test requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ local DSV4 ResidentTopKDriver integration test via cargo oxide (arch: {{ _cuda-arch }})"
    cargo oxide test --arch {{ _cuda-arch }} -- --features cuda,local-dsv4-tests --release -p ferrule-cli --test dsv4_resident_runtime_local -- --ignored --nocapture {{ args }}

dsv4-prefill-parity model='models/DeepSeek-V4-Flash-DSpark' prompt='Hello' layers='43' backend='cuda' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "-> DSV4 prefill parity harness (arch: {{ _cuda-arch }}, layers: {{ layers }}, backend: {{ backend }})"
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    ./target/release/ferrule deepseek-v4-prefill-parity {{ model }} -p "{{ prompt }}" --max-layers {{ layers }} --backend {{ backend }} {{ args }}

cuda:
    cargo run -p ferrule-cli -- cuda

inspect-weightpack path:
    cargo run -p ferrule-cli -- inspect-weightpack {{ path }}

expert-stream-smoke model layer='0' expert='0' *args='':
    cargo run -p ferrule-cli -- expert-stream-smoke {{ model }} --layer {{ layer }} --expert {{ expert }} {{ args }}

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

dsv4-cuda-generate-json prompt='Hello' tokens='4' chunk='4096' output='target/dsv4-generate.json' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ CUDA JSON run via cargo oxide build (arch: {{ _cuda-arch }}, output: {{ output }})"
    @mkdir -p target
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    ./target/release/ferrule deepseek-v4-generate models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --backend cuda --max-tokens {{ tokens }} --output-head-chunk-rows {{ chunk }} --json {{ args }} | tee {{ output }}

dsv4-cuda-moe-profile prompt='Hello' tokens='4' chunk='4096' output='target/dsv4-moe-profile.json' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ CUDA MoE timing JSON run (arch: {{ _cuda-arch }}, output: {{ output }})"
    @mkdir -p target
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    FERRULE_CUDA_MOE_TIMING=1 ./target/release/ferrule deepseek-v4-generate models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --backend cuda --max-tokens {{ tokens }} --output-head-chunk-rows {{ chunk }} --json {{ args }} | tee {{ output }}

dsv4-cuda-moe-ab prompt='Hello' tokens='4' chunk='4096' *args='':
    @if [ "{{ _use-cuda }}" != "1" ]; then echo "error: CUDA run requires cargo-oxide and an NVIDIA GPU (oxide={{ _has-oxide }}, gpu={{ _has-gpu }})"; exit 1; fi
    @echo "→ CUDA MoE TC/scalar A/B JSON timing (arch: {{ _cuda-arch }})"
    @mkdir -p target
    cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli
    FERRULE_CUDA_MOE_TIMING=1 FERRULE_CUDA_MOE_TC=1 ./target/release/ferrule deepseek-v4-generate models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --backend cuda --max-tokens {{ tokens }} --output-head-chunk-rows {{ chunk }} --json {{ args }} > target/dsv4-moe-tc.json
    FERRULE_CUDA_MOE_TIMING=1 FERRULE_CUDA_MOE_TC=0 ./target/release/ferrule deepseek-v4-generate models/DeepSeek-V4-Flash-DSpark --prompt "{{ prompt }}" --backend cuda --max-tokens {{ tokens }} --output-head-chunk-rows {{ chunk }} --json {{ args }} > target/dsv4-moe-scalar.json
    python3 -c 'import json; paths=[("tc","target/dsv4-moe-tc.json"),("scalar","target/dsv4-moe-scalar.json")]; [print("{}: decode_tok/s={:.3f} total={:.3f}s moe_calls={} tc/scalar/reduce={}/{}/{} moe_total={:.3f}s input={:.3f}s gate_up={:.3f}s swiglu={:.3f}s hidden_pack={:.3f}s down={:.3f}s".format(label,s["decode_tok_per_s"],d["total_seconds"],t["moe_calls"],t["moe_tc_calls"],t["moe_scalar_calls"],t["moe_reduce_calls"],t["moe_total_us"]/1e6,t["moe_input_prepare_us"]/1e6,t["moe_gate_up_us"]/1e6,t["moe_swiglu_us"]/1e6,t["moe_hidden_pack_us"]/1e6,t["moe_down_us"]/1e6)) for label,path in paths for d in [json.load(open(path))] for s in [d["summary"]] for t in [s["counters"]["timing"]]]; print("wrote target/dsv4-moe-tc.json and target/dsv4-moe-scalar.json")'



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
    @tokens="{{ tokens }}"; tokens="${tokens#tokens=}"; case "$tokens" in ''|*[!0-9]*) echo "error: dsv4-chat tokens must be an integer; use 'just dsv4-chat 64' or 'just dsv4-chat tokens=64'"; exit 2;; esac; echo "→ CUDA chat via cargo oxide build (arch: {{ _cuda-arch }}, tokens: $tokens)"; cargo oxide build --features cuda --arch {{ _cuda-arch }} -- --release -p ferrule-cli; ./target/release/ferrule chat models/DeepSeek-V4-Flash-DSpark -q cuda -n "$tokens" --chat-template deepseek-v4 --temp 0 {{ args }}

info model:
    cargo run --release -p ferrule-cli -- info {{ model }}

# ── Clean ──────────────────────────────────────────────────────────────

clean:
    cargo clean
    rm -f ./*.o ./*.ptx ./*.ll ./*.opt.ll ./*.cubin ./*.fatbin ./*.sass
