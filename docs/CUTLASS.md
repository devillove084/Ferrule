# CUDA/CUTLASS provider

Ferrule 的 CUDA kernel 实现位于 `crates/ferrule-backend`。CUTLASS/CuTe 是 CUDA
provider 的内部实现依赖，不是模型、调度器、materialization protocol 或跨硬件 ABI。

当前已验证 architecture and capability profile：

```text
compute capability 10.3
sm_103
CUDA 13.3
CUTLASS 4.6.1
```

设备 compute capability 被字面映射为公开 target；compute capability 10.3 映射为 `sm_103`。
完整指令集所需的 NVCC feature codegen 由 `build.rs` 在 provider 内部自动选择，不进入
设备 capability、operator ABI 或模型 plan。其他 target 使用同一个 capability/parser 和
provider-neutral plan，但必须分别通过 build/runtime 验收。

## Ownership boundary

Rust host 唯一拥有：

- CUDA context、stream、event、allocation 和 graph-stable workspace；
- provider-neutral executable model plan；
- paged KV、sequence 和 transaction lifecycle；
- materialization、expert residency、I/O 和调度；
- completion、error propagation 和 shutdown drain。

Native provider 只临时使用 Ferrule 传入的 device pointers、workspace 和 stream。C++
对象不跨 ABI；provider 不取得 transaction、KV、resource identity 或 scheduler ownership。

## Current organization

```text
crates/ferrule-backend/
  architecture_target.rs
  build.rs
  src/
    plan.rs
    cuda/
      runtime.rs
      architecture.rs
      provider.rs
      cutlass.rs
      context.rs
      ...
  native/cuda/
    abi/
      core_provider.h
      cutlass_provider.h
    implementations/
      portable/
        entrypoints.cu
      cutlass/
        entrypoints.cu
        architectures/
        capabilities/
```

`plan.rs` 定义 provider-neutral semantic requirements、capabilities、kernel identity 和
pre-resolved launch descriptors。Native `capabilities/` 按 semantic capability 组织实现，
`architectures/` 则包含 provider 私有的 architecture implementations。`cuda/provider.rs` 在
prepare/compile 阶段发现 native capabilities 并生成 model plan；hot path 不查询 architecture
string、环境变量或动态 trait。缺失 operation/capability 是 typed fatal error，不存在 CPU 或
另一 CUDA provider 的静默 fallback。

## Semantic provider ABI

当前 CUTLASS ABI 发布十个 semantic operations：

| Operation | Fused boundary |
|---|---|
| FP8 QueryA + KV | one packed activation producer, two projection consumers |
| BF16 compressor | one F32→BF16 activation tile, two projections |
| HC producer | HC mix/split → pre-RMSNorm → FP8/E8M0 pack |
| shared FFN | gate/up → SwiGLU → hidden pack → down |
| routed MXFP4 MoE | stable-frame resolve → gate/up → pack → down |
| MLA output | OutputA → BF16 latent boundary → FP8 pack → OutputB |
| Proposal attachment main projection/norm | target tap projection → boundary → RMSNorm |
| Proposal hybrid MLA attention | committed paged context + ephemeral proposal KV |
| Proposal head | HC head/final norm → LM projection → proposal selection |
| FP8 projection | one packed activation producer, one projection consumer |

Model code binds semantic roles, not row-count kernel variants. Small-M/tile schedule、architecture
specialization、MMA atom、workspace layout 和 launch geometry 都属于 provider 私有实现。

## Architecture and build

`crates/ferrule-backend/build.rs`：

1. 默认从 GPU 0 的 compute capability 自动得到 exact device target；
2. `FERRULE_CUDA_ARCH` 仅用于显式 cross-build override；
3. 验证 NVCC 发布对应 base code，并在 provider 内部选择所需 feature codegen；
4. 生成 device target/capability compile definitions；
5. 使用标准 Cargo `cc::Build` + NVCC 编译 portable/CUTLASS entrypoints 及其实现；
6. 链接 shared CUDA runtime。

Ferrule 不使用 `cargo-oxide`。CUTLASS 是 header-only pinned dependency：

```text
repository  https://github.com/NVIDIA/cutlass.git
tag         v4.6.1
commit      e05f953a5b3d38adc240df2ff928e0421c2abba3
```

准备依赖并构建当前 GPU：

```bash
just cutlass-setup
just build-cuda
```

显式构建 `sm_103` profile：

```bash
just build-cuda sm_103
```

直接使用 Cargo 时只需先准备 CUTLASS checkout；当前 GPU 会被自动检测：

```bash
cargo build --locked --release --features cuda
```

仅 cross-build 时显式覆盖 target：

```bash
FERRULE_CUDA_ARCH=sm_103 cargo build --locked --release --features cuda
```

可用 `FERRULE_CUTLASS_DIR` 指向另一个 checkout，但它必须满足 pinned version/manifest
contract。

## Validation contract

Provider operation 可用必须同时满足：

- provider ABI、CUTLASS version 和 native/Rust manifest 一致；
- compiled target 与 native provider target 一致；
- semantic operation、execution mode、dtype、layout 和 determinism capability 可满足；
- pointers、shape、workspace、stream 和 generation binding 合法；
- synchronous status 与 asynchronous status record 都成功；
- numerical、near-tie、route-order 和 transaction acceptance semantics 不漂移；
- complete model path 正确且端到端性能不回退。

当前 `sm_103` profile 已通过标准 release build 和 43 层 DeepSeek-V4 proposal attachment 固定生成正确性。
Backward/update catalog、更多 architecture 独立回归、完整 timeline 和 vLLM/SGLang 性能 gate
仍未完成。

## Benchmark rule

CUTLASS/Tensor Core 标签本身不是性能证据。每个 semantic operation 必须：

1. 测量完整 fused boundary，包括 pack、epilogue 和必要 synchronization；
2. 覆盖 small-M、batch/prefill 和 cross-tile shapes；
3. 报告 workspace bytes、steady-state allocations 和 stream/event behavior；
4. 验证 graph capture/replay 和 concurrent workspace isolation；
5. 通过 numerical、near-tie、route 和 proposal acceptance parity；
6. 只有完整 43 层 workload 改善时才接受 microbenchmark 优化。

另见 [Kernel Provider ABI](KERNEL_PROVIDER_ABI.md)、[Architecture](ARCHITECTURE.md) 和
[Roadmap](ROADMAP.md)。
