# DeepInfer provenance and architecture boundary

本记录是 Ferrule 的合规/架构边界说明，不构成法律意见。

## Reference and local review

- canonical URL（本地 clone 来源记录）：`https://github.com/deepseek-ai/DeepInfer`
- 当前参考 commit：`553741557a57df44a46fb3ab72b6aafeaf945337`；这是**本地浅克隆记录的参考commit**，来自本地 `target/DeepInfer`。
- 许可证：Apache-2.0；依据本地浅克隆审查的 `target/DeepInfer/LICENSE` 和 `target/DeepInfer/pyproject.toml`。其中 `pyproject.toml` 的 `project.license` 为 `Apache-2.0`，并以 `license-files = ["LICENSE"]` 指向许可证文件。

本记录不声明当前远程可解析性，不提供 commit 的远程链接。DeepInfer 的 vLLM-derived 文件和未初始化的 vendor 子模块是独立的来源单元；若未来涉及直接复制或分发，必须分别核对上游 commit、`NOTICE`、许可证和归属要求。DeepInfer 的顶层 Apache-2.0 记录不能替代这些检查。

## 本阶段的 artifact boundary

本次 DeepInfer 参考阶段未将 `target/DeepInfer` 源码、vendor 内容或其二进制加入 Ferrule 的 tracked 源码或发布物。`target/DeepInfer` 下既有或被 Git 忽略的 third-party artifacts 另按各自 provenance 管理；不能据此声称整个 checkout 没有 third-party。

Ferrule 实现是独立的 Rust clean-room adaptation，仅使用 behavior reference；本阶段未修改生产代码、`LICENSE` 或 `NOTICE`。本记录不据此作性能或支持承诺。

## Ferrule 已落地的抽象映射

下列 commit 是当前 Ferrule 历史中对应实现的本地提交；它们记录抽象契约或路由边界，不表示复制了 DeepInfer 实现：

| Ferrule 模块 | Ferrule commit | 抽象映射 |
| --- | --- | --- |
| `crates/ferrule-backend/src/operator_contract.rs` | `b2db1e3`, `ad8817a` | dtype、view、quantization 与 fusion stages 的 operator contract |
| `crates/ferrule-model/src/transformer/expert_parallel.rs` | `51953c0` | placement epoch 与 route metadata |
| `crates/ferrule-common/src/host_kv.rs` | `4c3c502` | Host-KV read plan |
| `crates/ferrule-common/src/topology.rs` | `880719a` | PP/TP mesh 与 attached EP |
| `crates/ferrule-runtime/src/parallel/expert_process.rs` | `b990a63` | process expert route envelope；见下方 numeric 矩阵 |

这些映射描述 Ferrule 自身的协议、状态和拒绝边界，不构成对上游实现、硬件路径或兼容性的等价性声明。

## Numeric 支持矩阵

按当前 `b990a63` 及其最终代码路径记录如下；低层协议存在不等于 production numeric composition 支持：

- **generic CPU versioned route-aware**：低层 generic CPU route-aware 路径已接通；generic CPU worker 通过 route metadata 校验 placement epoch、owner、logical expert 和 route identity。该事实不改变 production numeric process 的 boot admission 边界。
- **低层 NumericCommand route endpoint**：`NumericCommand::ComputeRoutes`/`RouteResults` endpoint 存在且有测试，承载显式 route metadata。这是 runtime 协议层事实，不是 production numeric process 的端到端支持承诺。
- **numeric fixed single-replica legacy**：production composition numeric process 仅支持 fixed single-replica placement，且 placement epoch 必须为 `0`；legacy `Compute` 适用于该边界。
- **numeric versioned single-replica route protocol**：低层 runtime protocol 存在并有测试，但 production composition numeric process boot 对所有 `epoch != 0`/versioned placement 拒绝。因此不能称为 versioned single-replica 端到端可用。
- **numeric replicated/nonzero-replica residency**：同样在 composition/boot 边界被拒绝；当前缺少独立 residency/provider authority。该项不是唯一拒绝边界，versioned placement 即使是 single-replica 也会因 `epoch != 0` 被拒绝。

因此，准确表述是：低层 route protocol 存在且有测试；production numeric process 只接受 fixed single-replica epoch 0。不能把低层 endpoint 的存在写成 production numeric versioned 支持，也不能把生产拒绝范围缩写为仅 nonzero replica。

## 不得称为 Ferrule 已实现的 DeepInfer 能力

以下能力不能据此称为 Ferrule 已实现：

- `DeepInfer`'s `DeepSeek-V4.1-Flash`-specific implementation/SM100-CUDA13 provider and its validation were not transplanted；其中包括其 DeepGEMM、DeepEP、FlashMLA provider paths。
- dynamic weight migration；
- full numeric replicated residency。

该边界不否认 Ferrule 自己的 DeepSeek adapter，也不否认仓库已有 GPU validation；它只是不把 DeepInfer 的特定实现、provider 或 validation 归为 Ferrule 已移植能力。本记录不作性能、硬件支持或生产兼容性承诺。

## 许可证待办

若未来直接复制标注为 `Derived from vLLM` 的文件或 vendor 内容，应在变更中核对并记录上游 commit、`NOTICE`、许可证和所需归属声明，并保留这些声明。当前没有将 `target/DeepInfer` 源码、vendor 内容或其二进制加入 Ferrule tracked 源码或发布物，因此不添加具体 `THIRD_PARTY` 版权归属，以免误称已分发第三方代码；target 下既有或被忽略的 third-party artifacts 仍按各自 provenance 管理。
